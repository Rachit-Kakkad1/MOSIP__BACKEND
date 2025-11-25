# backend/ocr/trocr_pipeline.py  (use this exact code)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

class TroCRPipeline:
    def __init__(self, model_name="microsoft/trocr-base-handwritten", device="cpu"):
        self.device = torch.device(device)
        # processor handles image preprocessing + tokenizer
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        # VisionEncoderDecoderModel is correct for TrOCR
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

    def predict(self, pil_image):
        # processor expects PIL image input
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_length=1024)
        decoded = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]

        # compute token-level confidences if scores present
        token_confidences = []
        try:
            import torch.nn.functional as F
            for step_scores in out.scores:
                probs = F.softmax(step_scores, dim=-1)
                top_prob = probs.max().item()
                token_confidences.append(top_prob)
            avg_conf = float(sum(token_confidences) / len(token_confidences)) if token_confidences else 0.0
        except Exception:
            avg_conf = 0.0

        return {"text": decoded, "score": float(avg_conf), "token_confidences": token_confidences}
