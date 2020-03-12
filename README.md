# Multimodal-short-video-classification

inceptionresnetv2.py实现了inceptionresnetv2网络，去掉最后的全连接层，输入为torch.Size([1, 3, 299, 299])，输出为torch.Size([1, 1536, 8, 8])。问题：不知道截取到哪一层，就只将最后一层全连接去掉。（1为batch_size，inceptionresnetv2模型下载地址：链接：https://pan.baidu.com/s/1jbREy49-wku1xYIYZJUkuQ 提取码：rva7）

netvlad.py将inceptionresnetv2.py输出特征图作为输入，输出num_clusters * dims维特征向量，也即64 * 1536（上面输出的通道数），然后接上一个全连接层输出维度为1024，最后输出为torch.Size([1, 1024])。（1为batch_size）

bert.py有一个函数get_tokens和一个TextNet类。get_token函数输入为经过增加[CLS]和[SEP]的纯文本以及bert预训练产生的tokenizer，输出为tokens、segments和input_masks作为文本网络的输入。TextNet有个参数code_length作为最终网络将文本生成的特征向量长度，这里定义为1024。最后输出维度为torch.Size([2, 1024])。（2表示输入文本为两句，bert模型下载地址：链接：https://pan.baidu.com/s/1yE09dpUh0NmaoqYoQLV4eg 提取码：hmg8）

lmf.py中LMF网络将图像和文本模态进行融合，输入图像（三维张量表示）、文本（纯文本形式加[CLS]和[SEP]）和tokenizer，输出维度为torch.Size([1, 1024])。（1表示batch_size）问题：output_dim和rank的选取。

hmc.py定义了一个HMC模块和相关的loss。HMC参数为feature_size=1024（前面lmf得到的特征向量）以及其他参数根据分类数目进行定义，输入为lmf的输出，输出为得到的L1和L2标签。hmc_loss计算损失函数。

