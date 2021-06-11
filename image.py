# import libraries
from PIL import Image
from numpy.core.numeric import indices
from torchvision import models, transforms
import matplotlib.pyplot as plt
import torch
import streamlit as st
from io import BytesIO, StringIO
from resnet import resnet_predict
from alexnet import alexnet_predict
from googlenet import googlenet_predict
from vgg16 import vgg16_predict
st.title('Deep Learning Image Classification')


def main():

    # st.info(__doc__)

    file = st.file_uploader('Upload file', type=['jpg', 'png'])
    show_file = st.empty()

    if not file:
        show_file.info('Please upload a image file')
        return

    content = file.getvalue()

    if isinstance(file, BytesIO):
        show_file.image(file)

    select = st.selectbox('Select your algorithm',
                          ('ResNet101', 'AlexNet', 'GoogleNet', 'VGG16'))

    if select == 'ResNet101':
        st.subheader('Welcome to ResNEt')

        resnet_label = resnet_predict(file)
        for i in resnet_label:
            st.write("Prediction ", i[0], ",   Score: ", i[1])

    if select == 'AlexNet':
        st.subheader('Welcome to AlexNet')
        alexnet_label = alexnet_predict(file)
        for i in alexnet_label:
            st.write("Prediction ", i[0], ",   Score: ", i[1])

    if select == 'GoogleNet':
        st.subheader('Welcome to GoogleNet')
        googlenet_label = googlenet_predict(file)
        for i in googlenet_label:
            st.write("Prediction ", i[0], ",   Score: ", i[1])

    if select == 'VGG16':
        st.subheader('Welcome to VGG16')
        vgg16_label = vgg16_predict(file)
        for i in vgg16_label:
            st.write("Prediction ", i[0], ",   Score: ", i[1])

    compare = st.selectbox('See comparisons',
                           ('Accuracy Comparison of Models', 'Inference Time Comparison', 'Model Size Comparison'))

    if compare == 'Accuracy Comparison of Models':
        image1 = Image.open('accuracy.jpg')
        st.image(image1)

    if compare == 'Inference Time Comparison':
        image2 = Image.open('run time.jpg')
        st.image(image2)

    if compare == 'Model Size Comparison':
        image3 = Image.open('model size.jpg')
        st.image(image3)


main()
