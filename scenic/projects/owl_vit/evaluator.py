r"""Run LVIS evaluation.

This script runs inference on a TFDS dataset (by default, the LVIS validation
set), writes the predictions to disk in the LVIS JSON format, and runs the LVIS
API evaluation on the files.

The ground-truth annotations must be supplied in the LVIS JSON format in the
local directory or at --annotations_path. The official annotations can be
obtained at https://www.lvisdataset.org/dataset.

The model is specified via --checkpoint_path and a --config matching the model.

See flag definitions in code for advanced settings.

Example command:
python evaluator.py \
  --alsologtostderr=true \
  --config=clip_b32 \
  --output_dir=/tmp/evaluator

"""

import collections
import functools
import json
import multiprocessing
import os
import re
import runpy
import tempfile
from prettytable import PrettyTable
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib
import zipfile
import contextlib

import skimage
from skimage import io as skimage_io
from absl import app
from absl import flags
from absl import logging
from clu import preprocess_spec
from flax import linen as nn
import jax
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
from lvis.eval import LVISEval
from lvis.lvis import LVIS
from lvis.results import LVISResults
from matplotlib import pyplot as plt
from PIL import Image
from scenic.projects.owl_vit.notebooks import inference
import ml_collections
import numpy as np
from pycocotools.coco import COCO
from scipy.special import expit as sigmoid
from pycocotools.cocoeval import COCOeval
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.preprocessing import image_ops
from scenic.projects.owl_vit.preprocessing import label_ops
from scenic.projects.owl_vit.preprocessing import modalities
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

LVIS_VAL_URL = 'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip'

COCO_METRIC_NAMES = [
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
]

_DEFAULT_ANNOTATIONS_PATH = '~/annotations/lvis_v1_val.json'

flags.DEFINE_string(
    'config',
    None,
    'Name of the config of the model to use for inference.',
    required=True)
flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Checkpoint path to use. Must match the model in the config.',
    required=True)
flags.DEFINE_string(
    'output_dir', None, 'Directory to write predictions to.', required=True)
flags.DEFINE_bool(
    'overwrite', False, 'Whether to overwrite existing results.')
flags.DEFINE_string(
    'tfds_name',
    'lvis',
    'TFDS name of the dataset to run inference on.')
flags.DEFINE_string('split', 'validation', 'Dataset split to run inference on.')
flags.DEFINE_string(
    'annotations_path',
    _DEFAULT_ANNOTATIONS_PATH,
    'Path to JSON file with ground-truth annotations in COCO/LVIS format. '
    'If it does not exist, the script will try to download it.')
flags.DEFINE_enum('data_format', 'lvis', ('lvis', 'coco'),
                  'Whether to use the LVIS or COCO API.')
flags.DEFINE_enum('platform', 'cpu', ('cpu', 'gpu', 'tpu'), 'JAX platform.')
flags.DEFINE_string(
    'tfds_data_dir', None,
    'TFDS data directory. If the dataset is not available in the directory, it '
    'will be downloaded.'
    )
flags.DEFINE_string(
    'tfds_download_dir', None,
    'TFDS download directory. Defaults to ~/tensorflow-datasets/downloads.')
flags.DEFINE_integer(
    'num_example_images_to_save', 10,
    'Number of example images with predictions to save.')
flags.DEFINE_integer(
    'label_shift', 1,
    'Value that will be added to the model output labels in the prediction '
    'JSON files. The model predictions are zero-indexed. COCO or LVIS use '
    'one-indexed labels, so label_shift should be 1 for these datasets. Set '
    'it to 0 for zero-indexed datasets.'
)
flags.DEFINE_float(
    'confidence_threshold', 0.1,
    'Threshold for setting a minimum confidence value'
)
flags.DEFINE_float(
    'iou_threshold', 0.4,
    'Threshold for setting the iou threshold value for counting a match between gts and dets'
)
flags.DEFINE_float(
    'nms_threshold', 0.4,
    'Threshold for setting nms threshold'
)
flags.DEFINE_string(
    'input_directory', None,
    'Directory where the images and annotation file are placed'
)

FLAGS = flags.FLAGS

_MIN_BOXES_TO_PLOT = 5
_PRED_BOX_PLOT_FACTOR = 3


Variables = nn.module.VariableDict
ModelInputs = Any
Predictions = Any


def get_dataset(tfds_name: str,
                split: str,
                input_size: int,
                tfds_data_dir: Optional[str] = None,
                tfds_download_dir: Optional[str] = None,
                data_format: str = 'lvis') -> Tuple[tf.data.Dataset, List[str]]:
  """Returns a tf.data.Dataset and class names."""
  builder = tfds.builder(tfds_name, data_dir=tfds_data_dir)
  builder.download_and_prepare(download_dir=tfds_download_dir)
  class_names = builder.info.features['objects']['label'].names
  ds = builder.as_dataset(split=split)
  if data_format == 'lvis':
    decoder = image_ops.DecodeLvisExample()
  elif data_format == 'coco':
    decoder = image_ops.DecodeCocoExample()
  else:
    raise ValueError(f'Unknown data format: {data_format}.')
  pp_fn = preprocess_spec.PreprocessFn([
      decoder,
      image_ops.Keep(
          [modalities.IMAGE, modalities.IMAGE_ID, modalities.ORIGINAL_SIZE])
  ], only_jax_types=True)
  num_devices = jax.device_count()
  return ds.map(pp_fn).batch(1).batch(num_devices), class_names


def tokenize_queries(tokenize: Callable[[str, int], List[int]],
                     queries: List[str],
                     prompt_template: str = '{}',
                     max_token_len: int = 16) -> List[List[int]]:
  """Tokenizes a sequence of query strings.

  Args:
    tokenize: Tokenization function.
    queries: List of strings to embed.
    prompt_template: String with '{}' placeholder to use as prompt template.
    max_token_len: If the query+prompt has more tokens than this, it will be
      truncated.

  Returns:
    A list of lists of tokens.
  """
  return [
      tokenize(
          label_ops._canonicalize_string_py(prompt_template.format(q)),  # pylint: disable=protected-access
          max_token_len) for q in queries
  ]


def get_embed_queries_fn(
    module: nn.Module,
    variables: Variables) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Get query embedding function.

  Args:
    module: OWL-ViT Flax module.
    variables: OWL-ViT variables.

  Returns:
    Jitted query embedding function.
  """

  @jax.jit
  def embed(queries):
    return module.apply(
        variables,
        text_queries=queries,
        train=False,
        method=module.text_embedder)

  return embed


def get_predict_fn(
    module: nn.Module,
    variables) -> Callable[[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
  """Get prediction function.

  Args:
    module: OWL-ViT Flax module.
    variables: OWL-ViT variables.

  Returns:
    Jitted predict function.
  """  
  image_embedder = jax.jit(
      functools.partial(
          module.apply, variables, train=False, method=module.image_embedder
      )
  )

  box_predictor = jax.jit(
      functools.partial(module.apply, variables, method=module.box_predictor)
  )

  class_predictor = jax.jit(
      functools.partial(module.apply, variables, method=module.class_predictor)
  )

  def apply(method, **kwargs):
    return module.apply(variables, **kwargs, method=method)
    
  def predict(images, query_embeddings):
    # Embed images:
    feature_map = image_embedder(images[None,...])
    b, h, w, d = feature_map.shape

    target_boxes = box_predictor(
        image_features=feature_map.reshape(b, h * w, d), feature_map=feature_map
    )['pred_boxes']

    out = class_predictor(
        image_features=feature_map.reshape(b, h * w, d),
        query_embeddings=query_embeddings[None, None, ...],  # [batch, queries, d]
    )

    return target_boxes, out

  return predict


@functools.partial(jax.vmap, in_axes=[0, 0, None, None])  # Map over images.
def get_top_k(
    scores: jnp.ndarray, boxes: jnp.ndarray, k: int,
    exclusive_classes: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Finds the top k scores and corresponding boxes within an image.

  The code applies on the image level; vmap is used for batching.

  Args:
    scores: [num_instances, num_classes] array of scores (i.e. logits or
      probabilities) to sort by.
    boxes: [num_instances, 4] Optional array of bounding boxes.
    k: Number of instances to return.
    exclusive_classes: If True, the top class for each box is returned. If
      False, classes are considered to be non-exclusive (multi-label setting),
      and the top-k computations happens globally across all scores, not just
      the maximum logit for each output token.

  Returns:
    Score, label, and box arrays of shape [top_k, ...] for the selected
    instances.
  """
  if scores.ndim != 2:
    raise ValueError('Expected scores to have shape [num_instances, '
                     f'num_classes], got {scores.shape}')

  if exclusive_classes:
    k = min(k, scores.shape[0])
    instance_top_scores = jnp.max(scores, axis=1)
    instance_class_ind = jnp.argmax(scores, axis=1)
    top_scores, instance_ind = jax.lax.top_k(instance_top_scores, k)
    class_ind = instance_class_ind[instance_ind]
  else:
    k = min(k, scores.size)
    top_scores, top_indices = jax.lax.top_k(scores.ravel(), k)
    instance_ind, class_ind = jnp.unravel_index(top_indices, scores.shape)

  return top_scores, class_ind, boxes[instance_ind]


def unpad_box(box_cxcywh: np.ndarray, *, image_w: int,
              image_h: int) -> np.ndarray:
  """Removes effect of ResizeWithPad-style padding from bounding boxes.

  Args:
    box_cxcywh: Bounding box in COCO format (cx, cy, w, h).
    image_w: Width of the original unpadded image in pixels.
    image_h: Height of the original unpadded image in pixels.

  Returns:
    Unpadded box.
  """
  padded_size = np.maximum(image_w, image_h)
  w_frac = image_w / padded_size
  h_frac = image_h / padded_size
  image_frac = np.array([w_frac, h_frac, w_frac, h_frac]) + 1e-6
  return np.clip(box_cxcywh / image_frac, 0.0, 1.0)


def format_predictions(*,
                       scores: np.ndarray,
                       labels: np.ndarray,
                       boxes: np.ndarray,
                       image_sizes: np.ndarray,
                       image_ids: np.ndarray,
                       label_shift: int = 0) -> List[Dict[str, Any]]:
  """Formats predictions to COCO annotation format.

  Args:
    scores: [num_images, num_instances] array of confidence scores.
    labels: [num_images, num_instances] array of label ids.
    boxes: [num_images, num_instances, 4] array of bounding boxes in relative
      COCO format (cx, cy, w, h).
    image_sizes: [num_images, 2] array of original unpadded image height and
      width in pixels.
    image_ids: COCO/LVIS image IDs.
    label_shift: Value that will be added to the model output labels in the
      prediction JSON files. The model predictions are zero-indexed. COCO or
      LVIS use one-indexed labels, so label_shift should be 1 for these
      datasets. Set it to 0 for zero-indexed datasets.

  Returns:
    List of dicts that can be saved as COCO/LVIS prediction JSON for evaluation.
  """
  predictions = []
  num_batches, num_instances = scores.shape

  for batch in range(num_batches):
    h, w = image_sizes[batch]
    for instance in range(num_instances):
      label = int(labels[batch, instance])

      score = float(scores[batch, instance])
      # Internally, we use center coordinates, but COCO uses corner coordinates:
      bcx, bcy, bw, bh = unpad_box(boxes[batch, instance], image_w=w, image_h=h)
      bx = bcx - bw / 2
      by = bcy - bh / 2
      predictions.append({
          'image_id': int(image_ids[batch]),
          'category_id': label + label_shift,
          'bbox': [float(bx * w), float(by * h), float(bw * w), float(bh * h)],
          'score': score
      })
  return predictions


def get_predictions(config: ml_collections.ConfigDict,
                    checkpoint_path: Optional[str],
                    tfds_name: str,
                    split: str,
                    top_k: int = 300,
                    exclusive_classes: bool = False,
                    label_shift: int = 0) -> List[Dict[str, Any]]:
  """Gets predictions from an OWL-ViT model for a whole TFDS dataset.

  These predictions can then be evaluated using the COCO/LVIS APIs.

  Args:
    config: Model config.
    checkpoint_path: Checkpoint path (overwrites the path in the model config).
    tfds_name: TFDS dataset to get predictions for.
    split: Dataset split to get predictions for.
    top_k: Number of predictions to retain per image.
    exclusive_classes: If True, the top class for each box is returned. If
      False, classes are considered to be non-exclusive (multi-label setting),
      and the top-k computations happens globally across all scores, not just
      the maximum logit for each output token.
    label_shift: Value that will be added to the model output labels in the
      prediction JSON files. The model predictions are zero-indexed. COCO or
      LVIS use one-indexed labels, so label_shift should be 1 for these
      datasets. Set it to 0 for zero-indexed datasets.

  Returns:
    Dictionary of predictions.
  """

  # Load model and variables:
  module = models.TextZeroShotDetectionModule(
      body_configs=config.model.body,
      normalize=config.model.normalize,
      box_bias=config.model.box_bias)

  config.init_from.checkpoint_path = checkpoint_path
  variables = module.load_variables(checkpoint_path=checkpoint_path)
  model = inference.Model(config, module, variables)
  predict = get_predict_fn(module, variables)
    
  # Create dataset:
  dataset, class_names = get_dataset(
      tfds_name=tfds_name,
      split=split,
      input_size=config.dataset_configs.input_size,
      tfds_data_dir=FLAGS.tfds_data_dir,
      tfds_download_dir=FLAGS.tfds_download_dir,
      data_format=FLAGS.data_format)

  input_images = load_input_json(FLAGS.input_directory + "/input_images.json")

  tokenized_queries = np.empty((0, 512))

  for input_image in input_images:
    path = "/content/input_images/images/" + input_image['path']
    image = Image.open(path)
    input_array = np.array(image)
    box = np.array(input_image['bbox'])
    tokenized_query, _ = model.embed_image_query(input_array, box)
    tokenized_queries = np.vstack([tokenized_queries, tokenized_query])

  # Prediction loop:
  predictions = []
  for batch in tqdm.tqdm(
      dataset.as_numpy_iterator(),
      desc='Inference progress',
      total=int(dataset.cardinality().numpy())):

    # Load input image:
    image = prepare_image(config, batch[modalities.IMAGE])

    target_boxes, out = predict(image, np.squeeze(tokenized_queries)) 

    logits = np.array(out['pred_logits'])[0, :, :]  # Remove padding.
    scores = sigmoid(np.max(logits, axis=-1))[0]
    labels = np.argmax(out['pred_logits'], axis=-1)[0][0]
    boxes = target_boxes[2]

    # Trova i valori che hanno scores che sono superiori alla soglia
    top_k_predictions = zip(scores, labels, boxes)
  
    # Effettuo NMS
    nms_boxes, nms_labels, nms_scores = nms(boxes, labels, scores, FLAGS.nms_threshold)

    # Converte le liste in array con una dimensione in più, richiesto per il codice
    nms_scores = np.array([list(nms_scores)])
    nms_labels = np.array([list(nms_labels)])
    nms_boxes = np.array([list(nms_boxes)])
    
    # Append predictions:
    predictions.extend(
        format_predictions(
            scores=nms_scores,
            labels=nms_labels,
            boxes=nms_boxes,
            image_sizes=batch[modalities.ORIGINAL_SIZE][0],
            image_ids=batch[modalities.IMAGE_ID],
            label_shift=label_shift))
  return predictions

def prepare_image(config, image):
  # Pad to square with gray pixels on bottom and right:
  image = np.squeeze(image)
  h, w, _ = image.shape
  size = max(h, w)
  image_padded = np.pad(
      image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5
  )

  # Resize to model input size:
  return skimage.transform.resize(
      image_padded,
      (config.dataset_configs.input_size, config.dataset_configs.input_size),
      anti_aliasing=True,
  )

def iou(box1, box2):
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x1, y1, x2, y2 = cx1 - w1/2, cy1 - h1/2, cx1 + w1/2, cy1 + h1/2
    x1_, y1_, x2_, y2_ = cx2 - w2/2, cy2 - h2/2, cx2 + w2/2, cy2 + h2/2

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)

    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def nms(boxes, labels, scores, iou_threshold=0.5):
    boxes = np.squeeze(boxes)
    labels = np.squeeze(labels)
    scores = np.squeeze(scores)

    if len(boxes) == 0:
        return [], [], []

    pick = []

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
        scores = np.array([scores])
        labels = np.array([labels])

    x1 = boxes[:, 0] - boxes[:, 2]/2
    y1 = boxes[:, 1] - boxes[:, 3]/2
    x2 = boxes[:, 0] + boxes[:, 2]/2
    y2 = boxes[:, 1] + boxes[:, 3]/2
   
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        suppress = [last]

        for pos in range(last):
            j = idxs[pos]
            if labels[i] == labels[j]:
                iou_val = iou(boxes[i], boxes[j])
                if iou_val > iou_threshold:
                    suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick], labels[pick], scores[pick]

def load_input_json(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            annotations = json_data.get("annotations", [])
            object_list = []
            for obj in annotations:
                object_list.append({
                    'id': obj.get('id', None),
                    'name': obj.get('name', None),
                    'path': obj.get('path', None),
                    'bbox': obj.get('bbox', None)
                })
        return object_list
    except Exception as e:
        print(f"Errore nel leggere il file JSON: {e}")
        return None

def _unshard_and_get(tree):
  tree_cpu = jax.device_get(tree)
  return jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), tree_cpu)


def write_predictions(predictions: List[Dict[str, Any]],
                      output_dir: str, split: str) -> str:
  filepath = os.path.join(output_dir, f'predictions_{split}.json')
  if tf.io.gfile.exists(filepath):
    raise ValueError(f'Output file already exists: {filepath}')
  with tf.io.gfile.GFile(filepath, 'w') as f:
    json.dump(predictions, f, indent=4)
  return filepath


def _download_file(url: str, path: str) -> None:
  """Downloads a file from a URL to a path."""
  logging.info('Downloading %s to %s', url, path)
  with tf.io.gfile.GFile(path, 'wb') as output:
    with urllib.request.urlopen(url) as source:
      loop = tqdm.tqdm(total=int(source.info().get('Content-Length')),
                       ncols=80, unit='iB', unit_scale=True, unit_divisor=1024)
      while True:
        buffer = source.read(8192)
        if not buffer:
          break
        output.write(buffer)
        loop.update(len(buffer))


def _download_annotations(annotations_path: str) -> str:
  """Downloads the appropriate annotations file."""
  filename = os.path.basename(annotations_path)
  if filename == 'lvis_v1_val.json':
    tf.io.gfile.makedirs(os.path.dirname(annotations_path))
    zip_path = annotations_path.replace('.json', '.zip')
    _download_file(url=LVIS_VAL_URL, path=zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
      f.extractall(os.path.dirname(annotations_path))
    tf.io.gfile.remove(zip_path)
  else:
    raise ValueError(f'Unknown annotations file: {filename}')

  return annotations_path

def calculate_ap(precision_scores, recall_scores):
    # Numero di intervalli di recall (0, 0.1, 0.2, ..., 1)
    recall_levels = np.arange(0, 1.1, 0.1)

    # Numero di classi
    num_classes = precision_scores.shape[1]

    # Inizializza un array per immagazzinare l'AP per ogni classe e ogni livello di recall
    ap_scores = np.zeros((len(recall_levels), num_classes))

    for i, recall_level in enumerate(recall_levels):
        for class_idx in range(num_classes):
            # Trova gli indici dove il livello di recall è maggiore o uguale a recall_level
            valid_recall_indices = np.where(recall_scores[:, class_idx] >= recall_level)[0]

            # Calcola la precisione massima per i livelli di recall validi
            max_precision = np.max(precision_scores[valid_recall_indices, class_idx]) if valid_recall_indices.size > 0 else 0

            # Memorizza il valore AP per questa classe e livello di recall
            ap_scores[i, class_idx] = max_precision

    return ap_scores

def run_evaluation(annotations_path: str,
                   predictions_path: str,
                   data_format: str = 'lvis') -> Dict[str, float]:
    """Runs evaluation and prints metric results."""
    with open(annotations_path, 'r') as file:
        coco_gt = json.load(file)

    with open(predictions_path) as file:
        coco_dt = json.load(file)

    # Define probability thresholds to use, between 0 and 1
    confidence_thresholds = np.linspace(0, 1, num=100)
    precision_scores = np.empty([len(confidence_thresholds), len(coco_gt.cats)])
    recall_scores = np.empty([len(confidence_thresholds), len(coco_gt.cats)])

    for c in confidence_thresholds:
      for ann_gt in coco_gt['annotations']:
          for ann_dt in coco_dt:
              # Se annotazione con confidenza minore di quella desiderata, skippo
              if ann_dt['score'] < c:
                  continue

              # Se annotazione o ground truth già matchata, skippo
              if 'matched' in ann_dt or 'matched' in ann_gt:
                  continue

              if (ann_gt['category_id'] != ann_dt['category_id'] or
                      ann_gt['image_id'] != ann_dt['image_id']):
                  continue

              iou = calculate_iou(ann_gt['bbox'], ann_dt['bbox'])

              if iou >= FLAGS.iou_threshold:
                  ann_gt['matched'] = True
                  ann_dt['matched'] = True

      coco_gt = COCO(annotations_path)

      # Calcolo i valori separati per categoria
      true_positive = [0] * len(coco_gt.cats)
      total_predictions = [0] * len(coco_gt.cats)
      total_ground_truth = [0] * len(coco_gt.cats)

      for ann_dt in coco_dt:
          total_predictions[ann_dt['category_id']] += 1
          if 'matched' in ann_dt:
              true_positive[ann_dt['category_id']] += 1

      for index, cat_id in enumerate(coco_gt.getCatIds()):
          gt_ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
          gt_anns = coco_gt.loadAnns(gt_ann_ids)
          gt_count = len(gt_anns)

          total_ground_truth[index] = gt_count

      for index, cat_id in enumerate(coco_gt.getCatIds()):
          if total_ground_truth[index] == 0:
            recall_scores[c][index] += round(0, 2)
          else:
            recall_scores[c][index] += round(true_positive[index] / total_ground_truth[index], 2)

          if total_predictions[index] == 0:
            precision_scores[c][index] += round(0, 2)
          else:
            precision_scores[c][index] += round(true_positive[index] / total_predictions[index], 2)

    # Calcolo degli AP scores
    ap_scores = calculate_ap(precision_scores, recall_scores)

    print_table(coco_gt, total_ground_truth, total_predictions, true_positive, ap_scores)

def print_table(coco_gt, total_ground_truth, total_predictions, true_positive, ap_scores):
    # Inizializza la tabella
    table = PrettyTable()
    table.field_names = ["class", "gts", "dets", "recall", "ap"]
    table.align["class"] = "l"
    table.align["gts"] = "l"
    table.align["dets"] = "l"
    table.align["recall"] = "l"
    table.align["ap"] = "l"

    # Ottieni gli ID delle categorie
    cat_ids = coco_gt.getCatIds()

    for index in range(len(cat_ids)):
        table.add_row([coco_gt.cats[index]['name'], total_ground_truth[index], total_predictions[index], ap_scores],
                      divider=index == len(cat_ids) - 1)

    mAP = round(mAP / len(coco_gt.cats), 2)
    table.add_row(["mAP", "", "", "", mAP])
    print(table)


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calcolo delle coordinate dell'intersezione
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    # Calcolo dell'area dell'intersezione
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # Calcolo dell'area di ciascun rettangolo
    area1 = w1 * h1
    area2 = w2 * h2

    # Calcolo dell'area dell'unione
    union_area = area1 + area2 - inter_area

    # Calcolo dell'IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def _set_host_device_count(n):
  xla_flags = os.getenv('XLA_FLAGS', '')
  xla_flags = re.sub(r'--xla_force_host_platform_device_count=\S+', '',
                     xla_flags).split()
  os.environ['XLA_FLAGS'] = ' '.join(
      ['--xla_force_host_platform_device_count={}'.format(n)] + xla_flags)


def plot_box(ax,
             ann,
             color,
             label=True,
             alpha=1.0,
             pad=3,
             labels=None,
             score=None):
  """Plots a single bounding box into axes."""
  x, y, w, h = ann['bbox']
  ax.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y],
          color=color,
          alpha=alpha)
  if label:
    s = str(ann['category_id'])
    if labels is not None and ann['category_id'] in labels:
      s = f"{ann['category_id']}: {labels[ann['category_id']]}"
    if score is not None:
      s = s + ' ' + f'{score:1.2f}'[1:]
    ax.text(
        x + pad,
        y + pad,
        s,
        ha='left',
        va='top',
        color=color,
        fontsize=10,
        fontweight='bold',
        alpha=alpha)


def plot_image(pixels, image_id, gt_by_image, pred_by_image, labels):
  """Plots an image with annotations."""
  fig, axs = plt.subplots(1, 2, figsize=(12, 6))

  # Plot ground-truth:
  ax = axs[0]
  ax.imshow(pixels)
  for ann in gt_by_image[image_id]:
    plot_box(ax, ann, color='g', labels=labels)
  ax.set_title(f'Ground truth (Image ID: {image_id})')

  # Plot prediction:
  ax = axs[1]
  ax.imshow(pixels)
  anns = pred_by_image[image_id]
  if anns:
    n = _MIN_BOXES_TO_PLOT + len(gt_by_image[image_id]) *  _PRED_BOX_PLOT_FACTOR
    n = min(n, len(anns))
    for ann in gt_by_image[image_id]:
      plot_box(ax, ann, color='g', label=False)
    for ann in anns:
      if ann['score'] <= FLAGS.confidence_threshold:
        continue
      plot_box(ax, ann, color='r', labels=labels, score=ann['score'])
  ax.set_title('Predictions')

  fig.tight_layout()
  return fig


def save_examples_images(*, ground_truth_path, pred_path, tfds_name, split,
                         output_dir, num_images, tfds_data_dir):
  """Saves example images to disk."""
  # Prepare annotations:
  with tf.io.gfile.GFile(ground_truth_path, 'r') as f:
    ground_truth = json.load(f)

  with tf.io.gfile.GFile(pred_path, 'r') as f:
    preds = json.load(f)

  gt_by_image = collections.defaultdict(list)
  for gt in ground_truth['annotations']:
    gt_by_image[gt['image_id']].append(gt)

  pred_by_image = collections.defaultdict(list)
  for pred in preds:
    pred_by_image[pred['image_id']].append(pred)

  labels = {cat['id']: cat['name'] for cat in ground_truth['categories']}

  images = list(
      tfds.load(
          tfds_name, split=split,
          data_dir=tfds_data_dir).take(num_images).as_numpy_iterator())

  # Plot and save images:
  file_names = []
  for image in images:
    image_id = image['image/id']
    fig = plot_image(image['image'], image_id, gt_by_image, pred_by_image,
                     labels)
    file_name = f'{image_id}.png'
    file_path = os.path.join(output_dir, file_name)
    with tf.io.gfile.GFile(file_path, 'wb') as f:
      fig.savefig(f, bbox_inches='tight')
    file_names.append(file_name)

  # Save index.html:
  with tf.io.gfile.GFile(os.path.join(output_dir, 'index.html'), 'w') as f:
    f.write('\n'.join([f'<img src="{n}" alt="{n}">' for n in file_names]))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make CPU cores visible as JAX devices:
  jax.config.update('jax_platform_name', FLAGS.platform)
  if FLAGS.platform == 'cpu':
    _set_host_device_count(max(1, multiprocessing.cpu_count() - 2))

  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  logging.info('JAX devices: %s', jax.device_count())

  # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  compilation_cache.initialize_cache('/tmp/jax_compilation_cache')

  config_name = os.path.splitext(os.path.basename(FLAGS.config))[0]
  output_dir = os.path.join(FLAGS.output_dir, config_name, FLAGS.tfds_name)
  tf.io.gfile.makedirs(output_dir)
  existing = tf.io.gfile.glob(os.path.join(output_dir, f'*_{FLAGS.split}.json'))
  if existing:
    if FLAGS.overwrite:
      for path in existing:
        tf.io.gfile.remove(path)
    else:
      print(
          f'Found existing results and --overwrite=false, exiting: {existing}')
      return

  if tf.io.gfile.exists(FLAGS.annotations_path):
    annotations_path = FLAGS.annotations_path
  else:
    annotations_path = _download_annotations(FLAGS.annotations_path)

  predictions = get_predictions(
      config=runpy.run_path(FLAGS.config)['get_config'](),
      checkpoint_path=FLAGS.checkpoint_path,
      tfds_name=FLAGS.tfds_name,
      split=FLAGS.split,
      label_shift=FLAGS.label_shift)

  logging.info('Writing predictions...')
  predictions_path = write_predictions(predictions, output_dir, FLAGS.split)

  logging.info('Running evaluation...')
  
  run_evaluation(annotations_path, predictions_path, # Qui se voglio provare meglio pycocotools
                             FLAGS.data_format)
  
  if FLAGS.num_example_images_to_save:
    logging.info('Saving example images...')
    examples_dir = os.path.join(output_dir, 'examples')
    tf.io.gfile.makedirs(examples_dir)
    save_examples_images(
        ground_truth_path=annotations_path,
        pred_path=predictions_path, # Qui se voglio provare meglio pycocotools, modifico il percorso con un file mio
        tfds_name=FLAGS.tfds_name,
        split=FLAGS.split,
        output_dir=examples_dir,
        num_images=FLAGS.num_example_images_to_save,
        tfds_data_dir=FLAGS.tfds_data_dir)

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)