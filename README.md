# Multimodal-short-video-classification

inceptionresnetv2.py实现了inceptionresnetv2网络，去掉最后的全连接层，输入为torch.Size([1, 3, 299, 299])，输出为torch.Size([1, 1536, 8, 8])。问题：不知道截取到哪一层，就只将最后一层全连接去掉。



短视频分类项目利用多模态融合进行实现。模态包含：视频（使用视频关键帧）、图像、音频和文本。图像使用ResNet-18模型进行特征提取，将最后全连接层输出的1000类别改为长度为hash_length作为最终输出的特征向量（详见model.py文件中ImageNet类）；因为视频VGGish网络提取音频的特征向量；文本使用BERT进行解析，每个句子输出code_length维特征向量（详见model.py文件中TextNet类）。
