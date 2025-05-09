import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import json
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") 

    def execute(self, requests):
        responses = []
        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, "QUERY_STRING").as_numpy()[0].decode('utf-8')
            inputs = self.tokenizer(query, return_tensors="np", padding="max_length", truncation=True, max_length=256)

            input_ids_tensor = pb_utils.Tensor("input_ids", inputs['input_ids'].astype(np.int64))
            attention_mask_tensor = pb_utils.Tensor("attention_mask", inputs['attention_mask'].astype(np.int64))

            inference_response = pb_utils.InferenceResponse(output_tensors=[input_ids_tensor, attention_mask_tensor])
            responses.append(inference_response)
        return responses