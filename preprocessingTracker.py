import numpy as np
import math
from PIL import Image, ImageEnhance
import cv2
import onnxruntime
import torch
class PreprocessingTracker():

    def __init__(self):
        #Using yolov5s for our purposes of object detection, you may use a larger model
        self.onnx_model_path = 'model640.onnx'
        self.onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
    def learning_preprocessing(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        
        resized_image = cv2.resize(img, (640, 640)).astype(np.float32) /255.0
        input_data = np.transpose(resized_image, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        output = self.onnx_session.run(None, {'input': input_data})
        prediction = np.squeeze(output[0])

        pred_mask = prediction
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1
        pred_mask = np.uint8(pred_mask * 255)
        resized_image = cv2.resize(img, (1920, 1080))
        resized_mask = cv2.resize(pred_mask, (1920, 1080))
        inpainted_image = cv2.inpaint(resized_image, resized_mask, 3, cv2.INPAINT_TELEA)
        fix_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        return fix_image

    def CV2_to_PIL_img(self, cv2_im):
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        return pil_im

    def PIL_to_CV2_img(self, img):
        cv_image = np.array(img.convert('RGB')) 
        cv_image = cv_image[:, :, ::-1].copy() 
        return cv_image

    def first_polynomial_function(self, image):
        table = np.array([1.657766*i-0.009157128*(i**2) + 0.00002579473*(i**3)
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def second_polynomial_function(self, image):
        table = np.array([
            -4.263256 * math.exp(-14)+1.546429*i-0.005558036*(i**2)+0.00001339286*(i**3)
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
                
        return cv2.LUT(image, table)

    def enhance_contrast(self, image, factor=1.4):
        _image = ImageEnhance.Contrast(
            self.CV2_to_PIL_img(image)
        ).enhance(factor)
        
        return self.PIL_to_CV2_img(_image)

    def reduce_glare(self, image):
        _image = self.adjust_gamma(
            self.second_polynomial_function(
                self.adjust_gamma(
                    self.first_polynomial_function(image), 
                    0.75
                )
            ),
            0.8
        )
        return _image

    def mix_filter(self, image):
        _image = self.enhance_contrast(
            self.reduce_glare(
                self.enhance_contrast(
                    self.reduce_glare(image), 
                    factor=1.6
                )
            ), 
            factor=1.4
        )
        return _image