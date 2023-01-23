import requests
import torch
from PIL import Image
import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from urllib3 import HTTPResponse
from urllib3._collections import HTTPHeaderDict

import open_clip
from open_clip.pretrained import download_pretrained_from_url


class DownloadPretrainedTests(unittest.TestCase):

    def create_response(self, data, status_code=200, content_type='application/octet-stream'):
        fp = BytesIO(data)
        headers = HTTPHeaderDict({
            'Content-Type': content_type,
            'Content-Length': str(len(data))
        })
        raw = HTTPResponse(fp, preload_content=False, headers=headers, status=status_code)
        return raw

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic_corrupted(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(b'corrupted pretrained model')
        with tempfile.TemporaryDirectory() as root:
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            with self.assertRaisesRegex(RuntimeError, r'checksum does not not match'):
                download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic_valid_cache(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            local_file = Path(root) / 'RN50.pt'
            local_file.write_bytes(file_contents)
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_not_called()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic_corrupted_cache(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            local_file = Path(root) / 'RN50.pt'
            local_file.write_bytes(b'corrupted pretrained model')
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_mlfoundations(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()[:8]
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            url = f'https://github.com/mlfoundations/download/v0.2-weights/rn50-quickgelu-{expected_hash}.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_mlfoundations_corrupted(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()[:8]
        urllib.request.urlopen.return_value = self.create_response(b'corrupted pretrained model')
        with tempfile.TemporaryDirectory() as root:
            url = f'https://github.com/mlfoundations/download/v0.2-weights/rn50-quickgelu-{expected_hash}.pt'
            with self.assertRaisesRegex(RuntimeError, r'checksum does not not match'):
                download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_hfh(self, urllib):
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:hf-internal-testing/tiny-open-clip-model')
        tokenizer = open_clip.get_tokenizer('hf-hub:hf-internal-testing/tiny-open-clip-model')
        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
        image = preprocess(Image.open(requests.get(img_url, stream=True).raw)).unsqueeze(0)
        text = tokenizer(["a diagram", "a dog", "a cat"])

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        self.assertTrue(torch.allclose(text_probs, torch.tensor([[0.0597, 0.6349, 0.3053]]), 1e-3))
