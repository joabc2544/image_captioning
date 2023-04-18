import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image

# กำหนด device ที่จะใช้ในการคำนวณของ PyTorch โดยจะเลือกใช้ GPU หากมี(device = "cuda") แต่หากไม่มีจะใช้ CPU แทน (device = "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# สร้างฟังก์ชัน caption_image_beam_search ที่รับ encoder, decoder, image_path, word_map และ beam_size เป็น input
def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size # สร่างตัวแปร k ที่รับค่าของ beam_size
    
    vocab_size = len(word_map) # สร่างตัวแปร vocab_size ที่รับค่าความยาวของตัวแปร word_map

    # Read image and process
    img = imread(image_path) # อ่านรูปภาพจาก image_path

    if len(img.shape) == 2: # ตรวจสอบรูปภาพที่อ่านว่าเป็น grayscale หรือไม่ (ถ้ารูปภาพมี shape == 2 แสดงว่าเป็น grayscale)

        img = img[:, :, np.newaxis] # หากรูปภาพเป็น grayscale บรรทัดนี้จะเพิ่ม axis ใหม่ให้กับรูปร่างของรูปภาพ
                                    # รูปร่างที่ได้จะเป็น (height, width, 1) (เลข 1 แสดงว่ารูปภาพมี channel เดียว (เช่น ระดับสีเทา)) 
   
        img = np.concatenate([img, img, img], axis=2) # หากรูปภาพเป็น grayscale บรรทัดนี้จะ concatenates ตาม axis ที่ 3 เพื่อให้เป็นภาพ 3 channel 
                                                      # ซึ่งการทำซ้ำ grayscale channel สามครั้ง รูปร่างที่ได้จะเป็น (height, width, 3) แสดงว่าภาพมี 3 channel (เช่น RGB)
        """concatenate() เป็นฟังก์ชัน NumPy ที่ใช้เชื่อมต่ออาร์เรย์ตั้งแต่สองอาร์เรย์ขึ้นไปตามแกนที่ระบุ"""

    img = imresize(img, (256, 256)) # บรรทัดนี้จะทำการปรับขนาดภาพเป็นสี่เหลี่ยมจัตุรัส 256x256 พิกเซลโดยใช้ฟังก์ชัน imresize()
    """imresize() ใช้เพื่อประขนาดของรูปให้เป็นขนาดที่ต้องการ"""

    img = img.transpose(2, 0, 1) # บรรทัดนี้จะทำการย้ายมิติของ array ให้ channel ของสีมาเป็นมิติที่ 1 และมิติ height และ width กลายเป็นมิติที่ 2 และ 3 ตามลำดับ
                                 # ตามค่าเริ่มต้น ฟังก์ชัน imread() อ่านรูปภาพในรูปแบบ HWC (height, width, channel)
                                 # แต่ PyTorch ใช้รูปแบบ CHW (channel, height, width) สำหรับการประมวลผลภาพ
    """transpose() ใช้ในการย้ายมิติของ array"""

    img = img / 255. # บรรทัดนี้สเกลค่าสีพิกเซลให้อยู่ระหว่าง 0 ถึง 1

    img = torch.FloatTensor(img).to(device) # convert image array ให้เป็น FloatTensor และย้ายไปยังอุปกรณ์ที่เหมาะสม (GPU ถ้าไม่มีก็จะใช้ CPU แทน)
    """FloatTensor() สร้าง tensor และส่งคืนในข้อมูลที่เป็น input ในข้อมูลประเภท 32-bit floating point"""
    """to() """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # บรรทัดนี้ทำการ normalize รูปภาพโดยการแปลงนี้จะทำให้คือเฉลี่ยเป็น [0.485, 0.456, 0.406]
                                     std=[0.229, 0.224, 0.225])  # และส่วนเบี่ยงเบนมาตรฐานเป็น [0.229, 0.224, 0.225]
    """Normalize() เป็นฟังก์ชันของ PyTorch ใช้เพื่อปรับ tensor ภาพให้เป็นมาตรฐานด้วยค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานสำหรับแต่ละช่องสี"""
    
    transform = transforms.Compose([normalize]) # บรรทัดนี่จะทำการสร้างลำดับของการแปลงเพื่อที่จะใช้กับรูปภาพในกรณีนี้ทำเพียงแค่ normalize อย่างเดียว
    """Compose() เป็นฟังก์ชันของ PyTorch ที่ช่วยให้เราสามารถรวมการแปลงหลายรายการเข้าด้วยกันเพื่อใช้กับข้อมูลอินพุต """
    
    image = transform(img)  # (3, 256, 256) # (height, width, number of channels)
                            # บรรทัดนี้จะทำการแปลงโดยทีจะใช้ลำดับก่อนหน้านี้คือการทำ normalize อย่างเดียว ผลทำให้ image tensor และมิติเป็น (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256) # (batch size, height, width, number of channels)
                                # บรรทัดนี้เพิ่มมิติพิเศษให้กับ image tensor ที่ตำแหน่งที่่ 0 ซึ่งสอดคล้องกับ batch size ในที่นี้จะเป็น 1
                                # สามารถใช้เลขอื่นเป็น batch size ได้ ซึ่งหมายถึงจำนวนตัวอย่าง (เช่น รูปภาพ ข้อความ ฯลฯ) ที่ถูกประมวลผลพร้อมกันในการส่งต่อ/ย้อนกลับใน neural network
    """unsqueeze() เป็นฟังก์ชันของ PyTorch ใช้เพื่อเพิ่มมิติใหม่ให้กับ tensor ในตำแหน่งที่ระบุ"""

    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
                                  # บรรทัดนี้จะทำการส่งรูปภาพผ่านโมเดล encoder ซึ่งจะทำการสร้าง tensor ที่มีรูปร่าง 4 มิติต่างผลลัพธ์ที่เขียนไว้ข้างบน
                                  # enc_image_size คือ ความสูงความกว้างของภาพที่ทำการ encode
                                  # encoder_dim คือ จำนวน channel ในรูปภาพที่ทำการ encode

    enc_image_size = encoder_out.size(1) # บรรทัดนี้ทำการแค่แยกค่า enc_image_size จาก tensor encoder_out ซึ่งจะใช้ในภายหลังในกระบวนการคำบรรยาย
    encoder_dim = encoder_out.size(3) # บรรทัดนี้ทำการแยกค่า encoder_dim จาก tensor encoder_out ซึ่งจะใช้ในภายหลังในกระบวนการคำบรรยาย
    """ size() ใช้ในการหาขนาดหรือรูปร่างของ tensor และจะทำการส่งกลับเป็นจำนวนเต็ม tuple ที่แสดงถึงขนาดหรือมิติของ tensor """

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
                                                        # บรรทัดนี้ทำการ Flattens ในส่วนของ encoded image ของโมเดลให้เป็น tensor 2 มิติที่มีขนาดตามข้างบน
    """
        โดยที่ viwe() ใช้ในการปรับรูปร่างของ tensor และส่งคืน tensor ใหม่พร้อมข้อมูลเดียวกันแต่มีรูปร่างต่างกัน
            ตัวอย่าง
                x = torch.randn(2, 3, 4)
                print(x)
                -------------------------------------------------------------
                result | tensor([[[-0.0746,  1.1667,  0.4842,  1.6181],
                       |          [ 0.2669, -1.3916, -0.4777, -0.8424],
                       |          [-0.3504,  1.2819, -0.4454,  0.1026]],
                       |  
                       |         [[ 0.8352, -0.1903, -0.1127, -1.4571],
                       |          [-1.2578, -0.3035, -0.0752,  0.7251],
                       |          [ 0.2336, -1.0277, -0.3916,  0.4408]]])
                -------------------------------------------------------------
                z = x.view(3, 8)
                print(z)
                ------------------------------------------------------------------------------------------------
                result | tensor([[-0.0746,  1.1667,  0.4842,  1.6181,  0.2669, -1.3916, -0.4777, -0.8424],
                       |         [-0.3504,  1.2819, -0.4454,  0.1026,  0.8352, -0.1903, -0.1127, -1.4571],
                       |         [-1.2578, -0.3035, -0.0752,  0.7251,  0.2336, -1.0277, -0.3916,  0.4408]]) 
                ------------------------------------------------------------------------------------------------              
    """
    """
        โดยที่ -1 ในมิติที่สองหมายความว่า PyTorch จะคำนวณขนาดของมิติที่สองโดยอัตโนมัติตามขนาดรวมของ tensor และมิติอื่นๆ
            ตัวอย่าง
                x = torch.randn(2, 3, 4)
                print(x.shape)  
                ------------------------------
                result | x.shape = (2, 3, 4)
                ------------------------------
                y = x.view(-1, 4)
                print(y.shape) 
                ------------------------------
                result | y.shape = (6, 4)
                ------------------------------
    """

    num_pixels = encoder_out.size(1) # บรรทัดนี้จะทำการตั้งค่าตัวแปร num_pixels เป็นขนาดของมิติที่ 2 ของ tensor encoder_out ซึ่งแสดงจำนวนตำแหน่งเชิงพื้นที่ของ encoded image
                                     # ค่านี้จะใช้ในกระบวนการ decode ในภายหลังเพื่อกำหนดขนาดของ attention mechanism

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) # (k, num_pixels, encoder_dim)
                                                                 # บรรทัดนี้จะทำการใช้ expand() เพื่อสร้างเทนเซอร์ที่ใหญ่ขึ้นซึ่งประกอบด้วยสำเนาของ tensor encoder_out ต้นฉบับหลายชุดตามมิติแรกเพื่อสร้างแบตช์ขนาด k
    """expand() สร้าง tensor ใหม่โดยการขยายมิติของ encoder_out ด้วยการทำซ้ำมิติแรก เช่น กรณีนี้ทำจะทำการทำซ้ำมิติแรก k ครั้ง"""

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
                                                                             # ทำการสร้าง tensor k_prev_words โดยที่มี k row และ 1 column โดยจาก utils.py word_map['<start>'] มี index เท่ากับ 5
                                                                             # จะได้ผลลัพธ์ประมาณนี้ tensor([[5],
                                                                             #                           [5],
                                                                             #                           [5]])
    """LongTensor() เป็นฟังก์ชันที่สร้าง tensor 64-bit integer (signed)"""

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1) 
                         # สร้าง tensor seqs เท่ากับ tensor k_prev_words

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
                                                 # สร้าง tensor top_k_scores ของรูปร่าง (k, 1) และค่าเริ่มต้นทั้งหมดเป็น 0 tensor นี้จะใช้เก็บคะแนน top k
    """zero() สร้าง tensor ที่มีขนาดเท่ากับ input และมีค่าเริ่มใน tensor เท่ากับ 0"""

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
                                                                              # สร้าง tensor seqs_alpha ที่มีรูปร่าง (k, 1, enc_image_size, enc_image_size) 
                                                                              # และค่าเริ่มต้นทั้งหมดเป็น 1 tensor นี้จะใช้เพื่อจัดเก็บ attention maps สำหรับลำดับ top k 
    """ones() สร้าง tensor ที่มีขนาดเท่ากับ input และมีค่าเริ่มใน tensor เท่ากับ 1"""

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list() # สร้าง list ว่าง complete_seqs ที่จะใช้เก็บ completed sequences

    complete_seqs_alpha = list() # สร้าง list ว่าง complete_seqs_alpha ที่จะใช้เก็บ attention maps สำหรับ completed sequences

    complete_seqs_scores = list() # สร้าง list ว่าง complete_seqs_scores ที่จะใช้เก็บ scoresสำหรับ completed sequences

    # Start decoding
    step = 1 # initializes ตัวแปร step เท่ากับ 1 ตัวแปรนี้จะใช้เพื่อติดตาม step การ decoder ปัจจุบัน

    h, c = decoder.init_hidden_state(encoder_out) # initializes hidden state และ cell state สำหรับ decoder LSTM โดยใช้ encoder_out เพราะมีข้อมูลเกี่ยวกับคุณสมบัติของภาพ
    """init_hidden_state() เป็นฟังก์ชั่นในคลาส DecoderWithAttention จากไฟล์ model.py ใช้เพื่อกำหนดค่าเริ่มต้นต่างๆ ของ hidden state และ cell state"""

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True: # สร้างการวนลูปที่จะทำงานไปเรื่อยๆ จนกว่าจะมีคำสั่งหยุดการทำงาน

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                                                                 # ใช้ embedding layer ของ decoder เพื่อรับ embeddings ที่ทำนายเอาก่อนหน้าซึ่งคือ k_prev_words ซึ่งมีรูปร่างเท่ากับ (k, 1)
                                                                 # decoder.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
                                                                 # squeeze(1) ใช้เพื่อลบมิติที่ 1 ออกทำให้มีรูปร่างเป็น (k, embed_dim)
        """embedding() คือการเรียกใช้ embedding layer ของ Pytorch จาก Embedding(vocab_size, embed_dim)"""
        """squeeze() ใช้ในการลบมิติโดยในที่นี้ใช้ในการลบมิติที 1"""

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                                                        # คำนวณ attention featurs (awe) และ attention weights (alpha) ตาม hidden state ก่อนหน้า(h)
        """attention() เป็นการคำนวณ attention ที่สร้างขึ้นจากคลาส Attention ใน model.py"""

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
                                                                # เปลี่ยน attention weights ให้เป็นรูปร่าง (k, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                                                   # ทำการคำนวณโดยการส่งผ่าน hidden state ก่อนหน้า ผ่าน output layer f_beta ของ decoder 
                                                   # และตามด้วยการเรียกใช้ Sugmoid function โดยที่ Sigmiod function นั้นมีค่าอยู่ระหว่าง 0 ถึง 1
                                                   # gating scalar gate ก็จะมีค่าอยู่ระหว่าง 0 ถึง 1 เหมือนกัน
        """sigmoid() self.sigmoid = nn.Sigmoid() ใช้เพื่อทำการเรียนใช้ Sugmoid function จากไฟล์ model.py"""
        """f_beta()  คือ self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate จากไฟล์ model.py"""

        awe = gate * awe # ทำ feature gating โดยที่ scalar gate จะควบคุมความเกี่ยวข้องของ attention features สำหรับขั้นตอนการ decode ปัจจุบัน

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                                                                                 # ในบรรทัดนี้จะทำการสร้างคำแบบคำต่อคำ ในการสนซ้ำแต่ละครั้งโดยที่จะทำการ decode คำถัดไปในลำดับ และสร้าง
                                                                                 # hidden state และ cell state ซึ่งจะใช้ในการวนลูป decode ในครั้งถัดไปเพื่อสร้างคำถัดไปในลำดับคำบรรยาย
                                                                                 # torch.cat([embeddings, awe], dim=1) ทำการเชื่อมต่อ embeddings ทีทำนายไว้ก่อนหน้านี้ และ 
                                                                                 # attention features ตามมิติที่ระบุ คือ 1 สิ่งนี้จะสร้าง tensor ใหม่ที่ใช้เป็น input สำหรับฟังก์ชัน decoder.decode_step()
                                                                                 # ฟังก์ชัน decoder.decode_step() นำ tensor ที่ต่อกันนี้พร้อม hidden state และ cell state ปัจจุบันเป็น input
                                                                                 # และทำการ decode single step ใน LSTM decoder โดยจะอัปเดต hidden state และ cell state
                                                                                 # ตาม input และการดำเนินการภายใน LSTM และส่งกลับ hidden state และ cell state ทีอัพเดทแล้ว
        """
            torch.cat() เป็นคำสั่งใน Pytorch ใช้ในการเชื่อมต่อ tensor ตามมิติที่ระบุ
                ตัวอย่าง 
                    a = torch.tensor([[1, 2], [3, 4]])
                    b = torch.tensor([[5, 6], [7, 8]])
                    c = torch.cat([a, b], dim=1)
                    print(c)
                    ----------------------------------
                    result | tensor([[1, 2, 5, 6],
                           |         [3, 4, 7, 8]])
                    ----------------------------------

                    a = torch.tensor([[1, 2], [3, 4]])
                    b = torch.tensor([[5, 6], [7, 8]])
                    c = torch.cat([a, b], dim=0)
                    print(c)
                    ----------------------------------
                    result | tensor([[1, 2],
                           |         [3, 4],
                           |         [5, 6],
                           |         [7, 8]])
                    ----------------------------------
        """                                                                
        scores = decoder.fc(h) # (s, vocab_size)
                               # คำนวณคะแนนสำหรับคำถัดไปที่ทำนายโดยใช้ fully connected layer fc ของ decoder (linear layer to find scores over vocabulary) จากไฟล์ model.py
                               # output h จากบรรทัดก่อนหน้าจะถูกป้อนเป็น input ไปยังเลเยอร์นี้ ผลลัพธ์จาก fully connected layer จะมีรูปร่างเป็น (s, vocab_size) 
                               # โดยที่ s คือขนาดแบทช์ และ vocab_size(len(word_map))

        scores = F.log_softmax(scores, dim=1) # normalizes scores โดยใช้ log softmax
        """log_softmax() ใช้กับเวกเตอร์คะแนนแต่ละตัวตามมิติที่ 1 ซึ่งเป็นมิติของคำศัพท์ เทนเซอร์ที่ได้จะมีรูปร่างเหมือนกับเทนเซอร์ input แต่ด้วยคะแนนที่ normalizes แล้ว 
            จึงแสดง log probabilities(ความน่าจะเป็นของล็อก)"""

        # Add the previous top-k scores top_k_scores to the current scores scores.
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
                                                          # ติดตามลำดับคะแนน top-k ที่น่าจะเป็นไปได้มากที่สุดที่โมเดลได้สร้างมาจนถึงตอนนี้
        """
            expand_as() ขยาย tensor ให้มีรูปร่างเหมือนกับสิ่งที่กำหนด
                ตัวอย่าง 
                    top_k_scores = torch.tensor([1.5, 2.0, 0.5]) shape = (k(k=3),)
                    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4], shape = (s, vocab_size)
                                           [0.2, 0.3, 0.4, 0.5],
                                           [0.3, 0.4, 0.5, 0.6]])
                    print(top_k_scores.expand_as(scores))
                    --------------------------------------------------------------------
                    result | tensor([[1.5000, 2.0000, 0.5000, 1.5000],
                           |         [1.5000, 2.0000, 0.5000, 1.5000],
                           |         [1.5000, 2.0000, 0.5000, 1.5000]])
                    --------------------------------------------------------------------       
                    print(top_k_scores.expand_as(scores) + scores)
                    --------------------------------------------------------------------
                    result | tensor([[2.6000, 2.7000, 2.8000, 2.9000],
                           |         [3.5000, 3.6000, 3.7000, 3.8000],
                           |         [1.0000, 1.1000, 1.2000, 1.3000]])
                    --------------------------------------------------------------------
        """         

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1: # ตรวจสอบว่านี่เป็นขั้นตอนแรกของการ decode หรือไม่
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                                                                          # ถ้าตรงตามเงื่อนไข จะต้องพิจารณาเฉพาะลำดับแรกของ scores เท่านั้น ในกรณีนี้ topk score 
                                                                          # และคำที่เกี่ยวข้องจะได้รับจากการเรียก "topk" ในแถวแรกของคะแนน (เช่น scores[0])
            """
                topk() เป็นฟังก์ชั่นที่ indices รับองคฺ์ประกอบของ top k จาก scores จากโมเดล
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                        top_k_scores คือ ตัวแปรตัวยึดตำแหน่ง
                        top_k_words คือ tensor ของ indices ที่สอดคล้องกับค่า top k ใน scores
                        k คือ จำนวนองค์ประกอบสูงสุดที่จะดึงข้อมูล
                        0 คือ มิติข้อมูลเพื่อค้นหาองค์ประกอบ top k กรณีนี้ เป็นมิติที่ 1
                        True คือ กำหนดว่าจะให้ส่งคืนค่า top k พร้อมกับ indices หรือไม่ กรณีนี้ ตั้งค่านี้เป็น True จะส่งคืนค่าทั้ง 2
                        True คือ กำหนดว่าจะให้เรียงลำดับค่า top k จากมากไปหาน้อยหรือไม่ กรณีนี้ ตั้งค่านี้เป็น True จะเรียงลำดับค่า top k จากมากไปน้อย เพื่อให้องค์ประกอบแรกใน output มีค่ามากที่สุด
            """
        else: # หากนี่ไม่ใช่ขั้นตอนแรกของการ decode
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
                                                                                # คะแนน tensor จะต้องถูกทำให้ flattened เพื่อให้สามารถเรียงลำดับคะแนนทั้งหมดและเลือกคะแนน k อันดับสูงสุดได้

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
                                                   # prev_word_inds คือ row index ของคำก่อนหน้าในลำดับ คำนวณจาก top_k_words / vocab_size(len(word_map))
        next_word_inds = top_k_words % vocab_size  # (s)
                                                   # next_word_inds คือ column index ของคำถัดไปในลำดับ คำนวณจาก top_k_words หารด้วย top_k_words % vocab_size(len(word_map))
                                                   
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                                                                                      # seqs คือ tensor ที่เก็บ indices ของคำที่ทำนายไว้สำหรับแต่ละลำดับในการทำ beam search 
                                                                                      # ในบรรทัดนี้จะเชื่อมต่อคำที่ทำนายไว้ก่อนหน้านี้(seqs[prev_word_inds]) กับ
                                                                                      # คำที่ทำนายใหม่(next_word_inds.unsqueeze(1)) เพื่อสร้างเทนเซอร์อัลฟาใหม่

        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)  # (s, step+1, enc_image_size, enc_image_size)
                                                                                                         # เก็บค่าอัลฟาสำหรับแต่ละลำดับและขอบเขตของภาพ
                                                                                                         # ในบรรทัดนี้จะเชื่อมต่อ alpha ก่อนหน้า (alpha[prev_word_inds]) กับ
                                                                                                         # ค่า alpha ใหม่ (alpha[prev_word_inds].unsqueeze(1)) เพื่อสร้างเทนเซอร์อัลฟาใหม่

        # Which sequences are incomplete (didn't reach <end>)? ระบุว่าลำดับใดที่ยังไม่สมบูรณ์ (เช่น ยังไม่ถึง end token '<end>') และลำดับใดที่เสร็จสมบูรณ์
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']] # เป็นรายการความเข้าใจที่วนลูปโดยใช้ tensor next_word_inds 
                                                                                                                  # โดยใช้ enumerate() เพื่อรับ index และค่าของคำถัดไปที่ทำนายแต่ละคำ
                                                                                                                  # จากนั้นจะตรวจสอบว่าคำถัดไปที่ทำนายไม่เท่ากับ index ของโทเค็น <end> 
                                                                                                                  # ใน word_map หรือไม่ถ้าจริงแสดงว่ายังไม่เสรฺ็จ  
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)) # คำนวณโดยการลบ incomplete_inds ออกจากชุดของ index ตั้งแต่ 0 ถึงความยาวของ next_word_inds

        # Set aside complete sequences
        if len(complete_inds) > 0: # ตรวจสอบเงื่อนไขว่ามีลำดับใดสมบูรณ์ในการทำ beam search หรือไม่
            complete_seqs.extend(seqs[complete_inds].tolist()) # extend complete_seqs ด้วยลำดับจาก seqs tensor ที่สอดคล้องกับ indices ใน complete_inds และแปลง tensors เป็น lists
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist()) # extend complete_seqs_alpha ด้วยลำดับจาก seqs_alpha tensor ที่สอดคล้องกับ indices ใน complete_inds และแปลง tensors เป็น lists
            complete_seqs_scores.extend(top_k_scores[complete_inds]) # extend complete_seqs_scores ด้วยลำดับจาก top_k_scores tensor ที่สอดคล้องกับ indices ใน complete_inds 
        """
            ตัวอย่าง extend()
                my_list = [1, 2, 3]
                new_list = [4, 5]
                my_list.extend(new_list)
                print(my_list)
                ------------------------------------------
                result | my_list = [1, 2, 3, 4, 5]
                ------------------------------------------
        """
        k -= len(complete_inds)  # reduce beam length accordingly
                                 # ลดความยาวของ beam(k) ตามจำนวนของลำดับที่สมบูรณ์ทำให้ไม่ต้องพิจารณาในขั้นตอนต่อๆ ไปของการทำ beam search
        # Proceed with incomplete sequences
        if k == 0: # หากความยาวของ beam(k) ลดลงเหลือ 0 
            break  # ให้ออกจากลูปเนื่องจากไม่มีลำดับให้ extend
        seqs = seqs[incomplete_inds] # อัพเดท seqs โดยเก็บเฉพาะลำดับที่ยังไม่สมบูรณ์
        seqs_alpha = seqs_alpha[incomplete_inds] # อัพเดท seqs_alpha ด้วยวิธีเดียวกัน
        h = h[prev_word_inds[incomplete_inds]] # อัปเดต hidden state ของ decoder โดยใช้ indices ของคำก่อนหน้าในลำดับที่ไม่สมบูรณ์
        c = c[prev_word_inds[incomplete_inds]] # อัปเดต cell state ของ decoder โดยใช้ indices ของคำก่อนหน้าในลำดับที่ไม่สมบูรณ์
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]] # อัพเดท tensor output encoder ด้วยวิธีเดียวกัน
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1) # อัพเดท tensor โดยเก็บเฉพาะคะแนนของลำดับที่ไม่สมบูรณ์ และยกเลิกการบีบเพื่อให้ตรงกับขนาดของเทนเซอร์ถัดไปที่อัพเดท
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1) # อัพเดท tensor k_prev_words โดยเก็บเฉพาะ indices ของคำถัดไปสำหรับลำดับที่ไม่สมบูรณ์ 
                                                                    # และยกเลิกการบีบให้ตรงกับขนาดของ tensor top_k_scores อัพเดท

        # Break if things have been going on too long
        if step > 50: # ตรวจสอบว่าขั้นตอนปัจจุบันเกินความยาวสูงสุดที่ 50 หรือไม่
            break # ถ้าใช่ ให้ออกจากลูปทันที
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores)) # ตั้งค่า i เป็น indices ของ complete_seqs_scores สูงสุดในรายการ complete_seqs_scores
    seq = complete_seqs[i] # ให้ seq เป็นลำดับ indices i ในรายการ complete_seqs
    alphas = complete_seqs_alpha[i] # ให้ alphas เป็นลำดับ indices i ในรายการ complete_seqs_alpha

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
