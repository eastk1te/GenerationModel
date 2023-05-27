Generative Adversarial Networks (GANs): CycleGAN은 GAN의 기본 개념과 구조를 기반으로 합니다. GAN은 실제와 가짜 샘플을 구별하는 생성자와 판별자로 구성되며, 생성자는 실제와 유사한 데이터를 생성하도록 학습하고, 판별자는 이를 판별합니다. 이러한 GAN의 개념은 CycleGAN에도 적용되었습니다.

Unpaired Image-to-Image Translation: CycleGAN은 기존의 이미지 변환 작업에서의 한계를 극복하기 위해 제안되었습니다. 기존에는 매칭된 데이터 쌍이 필요했지만, CycleGAN은 매칭되지 않은 데이터 쌍으로부터 이미지 간의 변환을 학습할 수 있습니다. 이와 관련된 연구로는 DualGAN, UNIT 등이 있습니다.

Cycle Consistency: CycleGAN은 이미지 변환을 위해 순방향 변환과 역방향 변환 사이의 일관성을 유지하기 위한 개념인 "Cycle Consistency"를 도입했습니다. 이는 생성된 이미지를 다시 원래 도메인으로 변환한 결과가 입력 이미지와 유사해야 한다는 제약을 가지는 것을 의미합니다.

Progressive Training: Progressive GAN (PGAN)과 같은 연구들은 생성 모델을 점진적으로 학습하는 방법을 제안했습니다. 이러한 접근 방식은 CycleGAN의 학습 과정을 개선하고 안정성을 향상시키는 데 도움이 되었습니다.


사전훈련된 모델 활용 VGGNet
직접 학습 시킨 모델 활용 CycleGAN

---
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

# 입력 이미지와 스타일 이미지 경로
input_image_path = 'input.jpg'
style_image_path = 'style.jpg'

# VGGNet 모델 로드
vgg_model = VGG19(weights='imagenet', include_top=False)

# 입력 이미지와 스타일 이미지 전처리
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

input_image = preprocess_image(input_image_path)
style_image = preprocess_image(style_image_path)

# 입력 이미지와 스타일 이미지의 특성 추출
def extract_features(image, model):
    features = model.predict(image)
    return features

input_features = extract_features(input_image, vgg_model)
style_features = extract_features(style_image, vgg_model)

# Gram Matrix 계산
def calculate_gram_matrix(features):
    reshaped_features = tf.reshape(features, [-1, features.shape[-1]])
    gram_matrix = tf.matmul(reshaped_features, reshaped_features, transpose_a=True)
    return gram_matrix

style_gram_matrix = calculate_gram_matrix(style_features)

# Loss 계산
def calculate_content_loss(input_features, target_features):
    loss = tf.reduce_mean(tf.square(input_features - target_features))
    return loss

def calculate_style_loss(input_gram_matrix, target_gram_matrix):
    loss = tf.reduce_mean(tf.square(input_gram_matrix - target_gram_matrix))
    return loss

content_loss = calculate_content_loss(input_features, target_features)
style_loss = calculate_style_loss(input_gram_matrix, style_gram_matrix)
total_loss = content_loss + style_loss

# Gradient Descent Optimization
optimizer = tf.optimizers.Adam(learning_rate=0.01)

def train_step(image, target_features, target_gram_matrix):
    with tf.GradientTape() as tape:
        input_features = extract_features(image, vgg_model)
        input_gram_matrix = calculate_gram_matrix(input_features)
        content_loss = calculate_content_loss(input_features, target_features)
        style_loss = calculate_style_loss(input_gram_matrix, target_gram_matrix)
        total_loss = content_loss + style_loss
    gradients = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))

# 초기 입력 이미지 설정
generated_image = tf.Variable(input_image, dtype=tf.float32)

# Style Transfer 수행
epochs = 100
for epoch in range(epochs):
    train_step(generated_image, style_features, style_gram_matrix)

# 최종 결과 이미지
result_image = generated_image.numpy().reshape((224, 224, 3))
```
---
<!-- 흑백사진 컬러고화질 사진으로 변환. -->
https://github.com/cszn/BSRGAN

stable-diffusion 오픈소스. 이미지 기반
waifu-diffusion 만화 
novelai


> # Motivation.

Stable Diffusion, NovelAI, DALL-E 등 text-to-image, image-to-image 등 image translation 모델들이 여러 매스컴들을 통해 대중화되는 단계를 눈으로 목격했다. 따라서, AI를 공부하는 사람으로서 내 프로필사진은 내가 직접 만든 모델로 한번 만들어 보고자 진행하게 되었다.

우선 진행에 앞서 Generation model과 GAN에 대해 배워보자.

GAN에서 파생된 여러 모델들 

DCGAN, cGAN, CycleGAN 순으로 적용해보기.
AnimeGAN, CartoonGAN과 같은 pre-trained model을 활용해보기.

NovelAI
Stable Diffusion model : https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

GAN에서 text to image와 같은 stable diffusion algorithm 이나
image to image 등 분야도 다르다

나는 원하는 이미지가 있고, 이를 다른 화풍으로 재현하고 싶기 때문에
image to image를 골랐다 
이 중에서 
pix to pix
cycleGAN
SPADE
U-Net 
등 여러가지가 있는데, 이중 cycleGAN이 적합하다고 파악Generation model

cycle GAN도  

https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch
https://github.com/TachibanaYoshino/AnimeGAN
https://velog.io/@danbibibi/AI-%EC%95%B1-%EA%B0%9C%EB%B0%9C-v2v-value-to-value-CycleGAN-%EB%AA%A8%EB%8D%B8


생성모델 -> GAN -> DCGAN 구현 -> CGAN 구현 -> Cycle 구현.



---
[DCGAN](https://deep-learning-study.tistory.com/642)







---






---

> ## CycleGAN

https://junyanz.github.io/CycleGAN/

> ## Introduction

CycleGAN
Unpaired Image to Image Translation using Cycle-Consistent Adversarial Networks 논문에 나온 것처럼 쌍으로 연결되지 않은 이미지 간 변환을 보여줌.

image to image

![image](https://www.tensorflow.org/static/tutorials/generative/images/cyclegan_model.png?hl=ko)



---

