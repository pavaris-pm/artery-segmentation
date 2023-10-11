from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_model():
    # load the tuned model
    model_name = 'tontokoton/segformer-b0-finetuned-v0'
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    tuned_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    return feature_extractor, tuned_model

# for overlaying color, can be adjusted !
def med_palette():
    """medical palette (self-defined) that maps each class to RGB values."""
    return [
        [0, 0, 0],
        [216, 82, 24],
        [255, 255, 0],
        [125, 46, 141],
        [118, 171, 47],
        [161, 19, 46],
        [255, 0, 0],
        [0, 128, 128],
    ]

# to overlay the prediction mask onto an image
def get_seg_overlay(image, seg):
  color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
  palette = np.array(med_palette())
  for label, color in enumerate(palette):
      color_seg[seg == label, :] = color

  # Show image + mask
  img = np.array(image) * 0.5 + color_seg * 0.5
  img = img.astype(np.uint8)

  return img


def segment(image):
    # inference time
    feature_extractor, tuned_model = get_model()
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = tuned_model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # overlay an image to the plot
    pred_img = get_seg_overlay(image, pred_seg)

    return pred_img