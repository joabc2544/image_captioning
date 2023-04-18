import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

# สร้างฟังชั่นในการสร้างไฟล์ Input สำหรับการนำไปใช้ในการเทรนโมเดล
def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'} # ตรวจสอบว่าชุดข้อมูลเป็นหนึ่งในสามชุดนี้หรือไม่ ถ้าไม่จะทำการรายงาน AssertionError หากเงื่อนไขเป็นเท็จ และโปรแกรมจะหยุดดำเนินการ
    """assert มักใช้ในการ Debug เพื่อตรวจหาข้อผิดพลาด และสมมุติฐานบางอย่างระหว่างการทำงานของโปรแกรม"""

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j: # อ่าน json file จาก path ที่ให้มาจาก karpathy_json_path ในโหมดการอ่าน('r')และสร้าง object "j"

        data = json.load(j) # โหลดข้อมูลจาก object "j" โดยใช้ json.load() และเก็บข้อมูลนั้นในตัวแปร "data"

    # Read image paths and captions for each image
    train_image_paths = []      #|
    train_image_captions = []   #|
    val_image_paths = []        #| สร้าง list ว่างเพื่อใช้ในการเก็บ path และ คำบรรยายภาพสำหรับ dataset ที่จะใช้ในการ Train, Validate และ Test
    val_image_captions = []     #|
    test_image_paths = []       #|
    test_image_captions = []    #|


    word_freq = Counter() # สร้างตัวแปร word_freq เป็นตัวแปร Counter()
    """คลาส Counter() เป็นคลาสจาก library collections ใช้เพื่อนับความถี่(จำนวน)ขององค์ประกอบต่างๆ และส่งคือผลลัพธ์เป็น dictionary"""

    for img in data['images']: # วนลูปแต่ละ ['images'](ชื่อรูป) ใน data ที่มาจากไฟล์ json

        captions = [] # สร้างตัวแปร captions เป็น list ว่างใช้สำหรับเก็บคำบรรยายของรูปภาพ

        for c in img['sentences']: # วนลูป['sentences'](คำบรรยาย) แต่ละประโยคที่อยู่ใน ['images'](ชื่อรูป) ปัจจุบัน

            # Update word frequency
            word_freq.update(c['tokens']) # อัพเดทการนับคำแต่ละคำในในคำบรรยายปัจจุบันไปที่ word_freq
            """update() ใช้กับ Python dictionary เพื่ออัพเดทจำนวนคำ และคำใหม่จาก argument หรือ dictionaryอื่น เข้าไปใน dictionary ที่ต้องการ"""

            if len(c['tokens']) <= max_len: # ตรวจสอบความยาวของ c['tokens'] น้อยกว่าหรือเท่ากับ max_len หรือไม่

                captions.append(c['tokens']) # ถ้าเข้าเงื่อนไขให้ทำการเพิ่ม c['tokens'] ปัจจุบันลงไปใน captions ที่เป็น list ที่สร้างเอาไว้

        if len(captions) == 0: # เป็นเงื่อนไขที่ใช้ตรวจสอบว่าไม่มีคำบรรยายที่ถูกต้องซึ่งมีความยาวน้อยกว่าหรือเท่ากับ max_len

            continue # ใช้ในการข้ามการวนลูปครั้งนั้นไปยังการวนลูปครั้งต่อไป
        """เงื่อนไขนี้มีไว้เพื่อที่จะใช้ในการจักการกับข้อมูลที่ไม่ต้องการ เพราะถ้าข้อมูลไม่ตรงไปตามเงื่อนไขที่ต้องการก็ไม่สมเหตุสมผลที่จะนำใช้ในการสร้างโมเดล 
        เพราะว่าในการสร้างโมเดลนั้นต้องการคำบรรยายอย่างน้อยหนึ่งประโยคต่อหนึ่งรูป เพื่อใช้ในการเทรน และประเมินตัวโมเดล"""

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(image_folder, img['filename'])
        # การเขียนแบบยาว กับ comment
        """
            if dataset == 'coco': # ถ้าการ dataset เป็น 'coco'

                path = os.path.join(image_folder, img['filepath'], img['filename'])
                # ถ้าตรงตามเงื่อนไขจะทำการสร้าง path โดยจะรวม image_folder(path เริ่มต้นที่รับมา) + img['filepath'](ที่แบ่งออกเป็น Train และ Val) + img['filename'](ชื่อไฟล์รูปภาพ)

            else: # ถ้าไม่ตรงตามเงื่อนไข
            
                path = os.path.join(image_folder, img['filename'])
                # ถ้าตรงตามเงื่อนไขจะทำการสร้าง path โดยจะรวม image_folder(path เริ่มต้นที่รับมา) + img['filename'](ชื่อไฟล์รูปภาพ)
        """

        if img['split'] in {'train', 'restval'}: # ตรวจสอบว่าในกลุ่มข้อความ ['split'] ใน img(['images'](ชื่อรูป)) นั้นเป็นประเภท 'train' หรือ 'restval' หรือไม่
            train_image_paths.append(path) # เพิ่มเส้นทางของรูปภาพที่ใช้ในการ train ไปยัง train_image_paths
            train_image_captions.append(captions) # เพิ่มคำบรรบยายของภาพลำดับนั้นไปยัง train_image_captions

        elif img['split'] in {'val'}: # ตรวจสอบว่าในกลุ่มข้อความ ['split'] ใน img(['images'](ชื่อรูป)) นั้นเป็นประเภท 'val' หรือไม่
            val_image_paths.append(path) # เพิ่มเส้นทางของรูปภาพที่ใช้ในการ validate ไปยัง val_image_paths
            val_image_captions.append(captions) # เพิ่มคำบรรบยายของภาพลำดับนั้นไปยัง val_image_captions

        elif img['split'] in {'test'}: # ตรวจสอบว่าในกลุ่มข้อความ ['split'] ใน img(['images'](ชื่อรูป)) นั้นเป็นประเภท 'test' หรือไม่
            test_image_paths.append(path) # เพิ่มเส้นทางของรูปภาพที่ใช้ในการ test ไปยัง test_image_paths   
            test_image_captions.append(captions) # เพิ่มคำบรรบยายของภาพลำดับนั้นไปยัง test_image_captions

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)  #| ตรวจสอบว่าข้อมูลที่นำเข้าไปใน image_paths และ image_captions มีจำนวนเท่ากันเพราะให้ถ้าไม่เท่ากันนั้นจะแสดงว่าข้อมูลที่เข้ามานั้นมีลำดับที่ไม่ตรงกัน
    assert len(val_image_paths) == len(val_image_captions)      #| เช่น เมื่อทำการเพิ่ม path ที่ 10 เข้าไปก็จะต้องเพิ่มคำบรรยายที่ 10 เข้าไปเหมือนกัน ถ้าไม่ตรงแสดงว่ามีการทำงานผิดพลาด
    assert len(test_image_paths) == len(test_image_captions)    #|

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq] # สร้าง word ที่ทำเก็บคำ โดยการเลือกคำที่เจอบ่อยกว่า min_word_freq จากใน word_freq
    """key() ใช้เพื่อรับรายการคีย์ทั้งหมดจาก dictionary"""

    word_map = {k: v + 1 for v, k in enumerate(words)} # สร้าง dictionary เพื่อที่จะเก็บ word map
    # ตัวอย่าง word map 
    """
        word_freq = {'the': 500, 'cat': 200, 'dog': 150, 'on': 100, 'mat': 80, 'in': 70, 'hat': 60}

        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
            -------------------------------------------------
            result| words = ['the', 'cat', 'dog']
            -------------------------------------------------

        word_map = {k: v + 1 for v, k in enumerate(words)}
            -------------------------------------------------
            result| word_map = {'the': 1, 'cat': 2, 'dog': 3}
            -------------------------------------------------
    """

    word_map['<unk>'] = len(word_map) + 1 # ทำการเพิ่ม <unk> เข้าไปใน ใน word map โดยทีให้เท่ากับ len(word_map) + 1
    """word_map = {'the': 1, 'cat': 2, 'dog': 3, '<unk>': 4}"""

    word_map['<start>'] = len(word_map) + 1 # ทำการเพิ่ม <start> เข้าไปใน ใน word map โดยทีให้เท่ากับ len(word_map) + 1
    """word_map = {'the': 1, 'cat': 2, 'dog': 3, '<unk>': 4, '<start>': 5}"""

    word_map['<end>'] = len(word_map) + 1 # ทำการเพิ่ม <end> เข้าไปใน ใน word map โดยทีให้เท่ากับ len(word_map) + 1
    """word_map = {'the': 1, 'cat': 2, 'dog': 3, '<unk>': 4, '<start>': 5, '<end>': 6}"""

    word_map['<pad>'] = 0 # ทำการเพิ่ม <end> เข้าไปใน ใน word map โดยทีให้เท่ากับ 0
    """word_map = {'the': 1, 'cat': 2, 'dog': 3, '<unk>': 4, '<start>': 5, '<end>': 6, '<pad>': 0}"""

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq' # ทำการสร้างตัวแปร str ที่จะใช้เป็นชื่อไฟล์สำหรับ output โดยประกอบด้วย 
                                                                                                                      # ชื่อชุดข้อมูล, จำนวนคำบรรยายทั้งหมด, และความถี่ของคำขั้นต่ำ

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j: # บรรทัดนี้ทำการเปิดไฟล์ json ในโหมดเขียนเป็น object "j"
        json.dump(word_map, j) # จะทำการนำ word_map มาเขียนลงในรูปแบบของไฟล์ .json
        """dump() ใช้ใน json โดยการทำให้ object เป็น serial ซึ่ง word_map นั้นเป็น dictionary ที่มีลักษณะการเก็บข้อมูลที่คล้ายกับ json ไฟล์"""                    
        
    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123) # ทำการเซ็ต Random seed ในการสุ่มจาก libary random
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'), #|
                                   (val_image_paths, val_image_captions, 'VAL'),       #| ทำการสุ่มตัวอย่างการคำบรรยายสำหรับแต่ละรูปภาพ เพราะ 1 รูปภาพมีคำบรรยายหลายประโยค
                                   (test_image_paths, test_image_captions, 'TEST')]:   #|

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h: # ทำการเปิดไฟล์ HDF5 ในโหมด append เป็น object "h"
                                                                                                             # โดยที่จะใช้ชื่อ (output_folder + _IMAGES_ + base_filename).hdf5
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image # ทำการสร้าง attribute ใน HDF5 เพื่อที่จะเก็บจำนวนคำบรรยายที่สุ่มมาจากตัวอย่างต่อ 1 รูปภาพ

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8') # สร้าง Dataset ใหม่ภายใน HDF5 เพื่อที่จะทำการเก็บภาพ โดยที่ขนาดของชุดข้อมูลคือ (จำนวนภาพ 3, 256, 256)
                                                                                            # แล้วใช่ประเภทข้อมูล คือ uint8
            """create_dataset() ใช้ในกาสร้างข้อมูลชดใหม่ภายในไฟล์ HDF5 ในที่นี้จะใช้ในการเก็บของมูลของรูปภาพ"""                                                

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = [] # สร้าง list ว่างชื่อว่า enc_captions 
            caplens = [] # สร้าง list ว่างชื่อว่า caplens

            for i, path in enumerate(tqdm(impaths)): # วนลูป path ของรูปภาพและคำบรรยาย
                """
                enumerate() ใช่ในการช่วยให้เราสามารถวนซ้ำ และส่งกลับเป็น (index, data)
                # ตัวอย่าง
                    fruits = ['apple', 'banana', 'cherry']
                    for index, fruit in enumerate(fruits):
                        print(index, fruit)
                        ------------------------------
                        result | 0 apple
                               | 1 banana
                               | 2 cherry
                        ------------------------------
                    for index, fruit in enumerate(fruits, start=1):
                        print(index, fruit)
                        ------------------------------
                        result | 1 apple
                               | 2 banana
                               | 3 cherry
                        ------------------------------
                tqdm() ใช่ในการแสดงแถบความคืบหน้าในการวนซ้ำ
                # ตัวอย่าง
                      0%|          | 0/5 [00:00<?, ?it/s]
                     60%|██████    | 3/5 [00:00<00:00, ?it/s]
                    100%|██████████| 5/5 [00:00<00:00, ?it/s]
                """
                
                # Sample captions
                if len(imcaps[i]) < captions_per_image: # ตรวจสอบว่าจำนวนคำบรรยายของรูปภาพปัจจุบัน น้อยกว่าจำนวนคำบรรยายต่อ 1 รูปภาพหรือไม่

                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))] # หากคำบรรยายต่อรูปมีไม่พอจะทำการสุ่มตัวอย่างเพิ่มเติมจากคำบรรยายที่มีอยู่จนกว่าจะครบจำนวนที่ต้องการ
                    """choice() เป็นฟังก์ชั่นจาก libary random ใช้ในการสุ่มตัวอย่างจากข้อมูลที่เราต้องการจะสุ่ม"""

                else: # ถ้าไม่ตรงเงื่อนไข
                    captions = sample(imcaps[i], k=captions_per_image) # หากคำบรรยามีจำนวนที่เพียงพอแล้วจะทำการสุมตัวอย่างในจำนวนที่ต้องการซึ่งกำหนดด้วย captions_per_image

                # Sanity check
                assert len(captions) == captions_per_image # ตรวจสอบว่าจำนวนคำบรรยายสำหรับภาพปัจจุบันเท่ากับคำบรรยายที่ต้องการต่อ 1 ภาพ

                # Read images
                img = imread(impaths[i]) # ทำการอ่านไฟล์รูปจาก path ที่เก็บไว้ที่ impath ลำดับที่ i

                if len(img.shape) == 2: # ตรวจสอบรูปภาพที่อ่านว่าเป็น grayscale หรือไม่ (ถ้ารูปภาพมี shape == 2 แสดงว่าเป็น grayscale)

                    img = img[:, :, np.newaxis] # หากรูปภาพเป็น grayscale บรรทัดนี้จะเพิ่ม axis ใหม่ให้กับรูปร่างของรูปภาพ
                                                # รูปร่างที่ได้จะเป็น (height, width, 1) (เลข 1 แสดงว่ารูปภาพมี channel เดียว (เช่น ระดับสีเทา)) 

                    img = np.concatenate([img, img, img], axis=2) # หากรูปภาพเป็น grayscale บรรทัดนี้จะ concatenates ตาม axis ที่ 3 เพื่อให้เป็นภาพ 3 channel 
                                                                  # ซึ่งการทำซ้ำ grayscale channel สามครั้ง รูปร่างที่ได้จะเป็น (height, width, 3) แสดงว่าภาพมี 3 channel (เช่น RGB)


                    img = imresize(img, (256, 256)) # บรรทัดนี้จะทำการปรับขนาดภาพเป็นสี่เหลี่ยมจัตุรัส 256x256 พิกเซลโดยใช้ฟังก์ชัน imresize()
                    """imresize() ใช้เพื่อประขนาดของรูปให้เป็นขนาดที่ต้องการ"""

                    img = img.transpose(2, 0, 1) # บรรทัดนี้จะทำการย้ายมิติของ array ให้ channel ของสีมาเป็นมิติที่ 1 และมิติ height และ width กลายเป็นมิติที่ 2 และ 3 ตามลำดับ
                                                 # ตามค่าเริ่มต้น ฟังก์ชัน imread() อ่านรูปภาพในรูปแบบ HWC (height, width, channel)
                                                 # แต่ PyTorch ใช้รูปแบบ CHW (channel, height, width) สำหรับการประมวลผลภาพ
                    """transpose() ใช้ในการย้ายมิติของ array"""


                assert img.shape == (3, 256, 256) # ตรวจสอบว่ารูปภาพมี shape เท่ากับ (3, 256, 256) หรือไม่

                assert np.max(img) <= 255 # ตรวจสอบว่าค่าของ pixel ทั้งหมดที่อยู่ในรูปนั้นมีค่าที่อยู่ในช่วง 255 หรือไม่

                # Save image to HDF5 file
                images[i] = img # ทำการบันทึก array ของรูปภาพไปใน HDF5 ที่ index ที่ i

                for j, c in enumerate(captions): # วนคำบรรยายแต่ละรายการของรูปภาพ 1 รูป

                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c)) # ทำการสร้าง enc_c โดยเริ่มด้วย '<start>' + คำที่อยู่ในคำบรรยายที่มีคำใน word_map + <end> + <pad>(ถ้าความยาวของคำบรรยายสั้นเกิน)
                    """get() คึอ ทำการส่งกลับค่าสำหรับ(คำ)ที่ระบุใน dictionary หากไม่พบคีย์(คำ)ใน dictionary ระบบจะส่งกลับค่า default ซึ่งค่า default ในที่นี้คือ <unk>

                        my_dict = {'apple': 1, 'banana': 2, 'cherry': 3, 'the': 4, 'cat': 5, 'dog': 6}
                        max_len = 10
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in my_dict] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(my_dict)) 
                            ---------------------------------------------------------
                            result | enc_c = [5, 4, 4, 4, 1, 2, 3, 8, 0, 0, 0, 0]
                            ---------------------------------------------------------
                    """

                    # Find caption lengths
                    c_len = len(c) + 2 # สร้าง c_len เท่ากับ len(c) + 2 จาก <start> กับ <end>

                    enc_captions.append(enc_c) # ทำการเพิ่ม enc_c เข้าไปใน enc_captions
                    caplens.append(c_len) # ทำการเพิ่ม c_len เข้าไปใน caplens

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens) # คำนวณจำนวนคำบรรยาย จากจำนวนภาพและจำนวนคำบรรยายต่อภาพ ถ้าไม่ตรงกับ enc_captions และ caplens โปรแกรมจะหยุดทำงาน

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j: # เปิดไฟล์ JSON เพื่อบันทึกคำบรรยายที่ผ่านการ encode โดยที path เป็น 
                                                                                                              # (output_folder(path) + split(TRAIN, VAL หรือ TEST) + "_CAPTIONS_" + base_filename).json
                json.dump(enc_captions, j) # บรรทัดนี้จะบันทึกคำบรรยายที่ผ่านการ encode ลงในไฟล์ JSON ที่เปิดในบรรทัดก่อนหน้า

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j: # เปิดไฟล์ JSON เพื่อบันทึกความยาวของคำบรรยายที่ผ่านการ encode โดยที path เป็น 
                                                                                                             # (output_folder(path) + split(TRAIN, VAL หรือ TEST) + "_CAPLENS_" + base_filename).json
                json.dump(caplens, j) # บรรทัดนี้จะบันทึกความยาวของคำบรรยายที่ผ่านการ encode ลงในไฟล์ JSON ที่เปิดในบรรทัดก่อนหน้า


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1)) # บรรทัดนี้จะคำนวณ bias ที่จะใช่ในการคำนวณ weight เริ่มต้น embedding layer 
    """
        โดยการทำคำนวนนั้นจะทำการนั้นจะคำนวณด้วย รากที่ 3 ของ มิติที่ 2 ของ embedding layer 
        อ้างอิงจาก "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013).
    """
    torch.nn.init.uniform_(embeddings, -bias, bias) # ทำการ initializes weight ของ embedding tensor โดยใช้ค่าที่มาจาก uniform distribution ระหว่าง -bias กับ bias
    """torch.nn.init.uniform_() ใช้้เพื่อกำหนดค่าเริ่มต้นของ tensor จาก uniform distribution"""


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f: # ทำการเปิดไฟล์ emb_file ในโหมดอ่านและแทนด้วย object "f"

        emb_dim = len(f.readline().split(' ')) - 1 # สร้าง emb_dim เท่ากับการอ่านบรรทัดแรก และทำการแยกองต์ประกอบจากการช่องว่าง(split(" ")) 
                                                   # และทำการนับจำนวนองค์ประกอบจากผลลัพธ์ที่ทำการแยกองค์ประกอบซึ่งจะประกอบด้วย มิติของ embedding และค่าของแต่ละ embedding
                                                   # และเพราะว่าองค์ประกอบแรกของ embedding นั้นคือ คำทำให้เราทำการ -1  จากการนับจำนวนองค์ประกอบทั้งหมด
        """ตัวอย่าง embedding "apple 0.1 0.2 0.3 0.4 0.5" """

    vocab = set(word_map.keys()) # สร้าง vocab ให้เป็น set ที่เก็บ คำขอบ word_map
    """set() คือการสร้างตัวแปรที่จะเก็บองค์ประกอบทีไม่ซ้ำกัน แบบไม่มีลำดับ"""

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim) # สร้าง embeddings tensor ที่มีมิติ เท่ากับ (len(vocab) x emb_dim)

    init_embedding(embeddings) # ทำการ initailize embedding tensor ด้วยฟังก์ชั่น init_embedding จากข้างบน

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'): # ทำการเปิดไฟล์ emb_file และทำการวงลูปเพื่อที่จะอ่านที่ละบรรทัด
        line = line.split(' ') # ทำการแยกองค์ประกอบของแต่ละบรรทัดจากช่องว่าง

        emb_word = line[0] # สร้าง emb_word เพื่อที่จะเก็บคำที่มาจากไฟล์ emb_file
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:]))) # ทำการแ
        """
        lambda ใช้ในการกำหนดฟังก์ชั่นที่ไม่ถูกกำหนมาก่อน ในกรณีนี้ใช้เพื่อกำหนดฟังก์ชันที่แปลงแต่ละองค์ประกอบของการวนลูปแต่ละบรรทัด(line[1:])เป็นทศนิยม
        isspace() เป็นฟังก์ชันในตัวใน Python ที่คืนค่า True หากสตริงที่กำหนดประกอบด้วยช่องว่างทั้งหมด (เช่น ช่องว่าง แท็บ การขึ้นบรรทัดใหม่) 
            ไม่งั้นจะเป็น Falseในกรณีนี้ จะใช้ในการกรองอักขระช่องว่างออกจากวนลูปแต่ละบรรทัด(line[1:])
        filter() เป็นฟังก์ชันที่กำหนดส่งคืนค่าองค์ประกอบที่เป็น True ในกรณีนี้ถูกใช้เพื่อลบอักขระช่องว่างออกจากบรรทัด  
        map() ใช้กับแต่ละองค์ประกอบและส่งกลับแต่ละองค์ประกอบใหม่พร้อมผลลัพธ์ ในกรณีนี้ใช้ฟังก์ชันแลมบ์ดาที่แปลงแต่ละองค์ประกอบจากวนลูปแต่ละบรรทัด(line[1:]) 
            เป็นทศนิยม และส่งกลับแต่ละองค์ประกอบใหม่พร้อมผลลัพธ์

        ตัวอย่าง
            line = "0.123 0.456 0.789\n"
            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
                result | embedding = [0.123, 0.456, 0.789]
        """
        # Ignore word if not in train_vocab
        if emb_word not in vocab: # ถ้า emb_word ไม่อยู่ใน vocab
            continue # ข้ามคำการวนลูปของรอบนั้นไป

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding) # บรรทัดนี้กำหนด embedding vector emb_word ให้กับแถวที่สอดคล้องกันของ embedding tensor
                                                                      # โดยการเข้าถึง index ของคำใน word_map[emb_word]
                                                                      # จากนั้น embedding tensor จะถูกตั้งค่าให้เป็น PyTorch tensor ที่มี embedding vector
                                                                      # ซึ่งแปลง list ของ float โดยใช้ torch.FloatTensor(embedding)

    return embeddings, emb_dim # ส่งกลับ embeddings tensor กับ emb_dim(มิติของ มิติของ embedding) 


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups: # วนลูป optimizer.param_groups
        """
            param_groups เป็น attribute ใน optimizer ซึ่งคือ list ของ dictionary โดยที่ dictionary แต่ละรายการแสดงถึงกลุ่มของพารามิเตอร์ที่ต่างกัน
                dictionary แต่ละรายการในรายการ param_groups มีดังนี้คือ:
                    - 'params'คือ list ของพารามิเตอร์ที่เป็นของกลุ่มนี้
                    - 'lr'คือ learning rate ที่จะใช้สำหรับพารามิเตอร์กลุ่มนี้
                    - 'weight_decay'คือ L2 penalty ที่จะใช้กับพารามิเตอร์กลุ่มนี้ (ถ้ามี)
                    - 'momentum'คือ momentum factor ที่จะใช้สำหรับพารามิเตอร์กลุ่มนี้ (ถ้ามี)
                    - 'dampening'คือ dampening factor สำหรับ momentum (ถ้ามี)
                    - 'nesterov'คือ จะใช้ Nesterov momentum สำหรับพารามิเตอร์กลุ่มนี้หรือไม่ (ถ้ามี)
        """
        for param in group['params']: # วนซ้ำ group['params'] ปัจจุบัน
            if param.grad is not None: # ตรวจสอบว่ามีการไล่ระดับสีสำหรับพารามิเตอร์ปัจจุบันหรือไม่ และไม่ใช่ None
                param.grad.data.clamp_(-grad_clip, grad_clip) # ใช้การตัดไล่ระดับสีโดยยึดค่าของเทนเซอร์ไล่ระดับสีให้อยู่ในช่วง [-grad_clip, grad_clip]


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,  #| สร้างสถานะของ dictionary ที่มีข้อมูลที่จะบันทึกใน checkpoint รวมถึงหมายเลข epoch ปัจจุบัน 
             'bleu-4': bleu4,                                       #| จำนวน epochs ตั้งแต่การปรับปรุงครั้งล่าสุดโดย BLEU-4 score 
             'encoder': encoder,                                    #| คะแนนการตรวจสอบความถูกต้องของ BLEU-4 score 
             'decoder': decoder,                                    #| โมเดลตัวเข้ารหัสและตัวถอดรหัส และเครื่องมือเพิ่มประสิทธิภาพตามลำดับ
             'encoder_optimizer': encoder_optimizer,                #|
             'decoder_optimizer': decoder_optimizer}                #|
    
    filename = 'checkpoint_' + data_name + '.pth.tar' # สร้างชื่อไฟล์สำหรับ checkpoint ('checkpoint_' + data_name + '.pth.tar')

    torch.save(state, filename) # บันทึก dictionary สถานะ เป็นไฟล์ checkpoint ด้วยชื่อไฟล์ที่สร้างขึ้นในขั้นตอนก่อนหน้าโดยใช้ฟังก์ชัน torch.save()

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best: # หาก checkpoint นี้ดีที่สุดแล้ว
        torch.save(state, 'BEST_' + filename) # ให้บันทึกสำเนาของ ictionary สถานะ อีกชุดหนึ่งโดยเพิ่มคำนำหน้า "BEST_" ในชื่อไฟล์ เพื่อไม่ให้จุดตรวจที่แย่กว่านั้นเขียนทับ


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups: # วนลูป optimizer.param_groups
        param_group['lr'] = param_group['lr'] * shrink_factor # learning rate คูณด้วย shrink_factor เพื่อปรับตัวของ learning rate
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0) # สร้างตัวแปร batch_size เพื่อเก็บขนาดของมิติแรกของ tensor ซึ่งแสดงถึงขนาดของ batch

    _, ind = scores.topk(k, 1, True, True) # คำนวณค่า top-k
    """
        topk() เป็นฟังก์ชั่นที่ indices รับองคฺ์ประกอบของ top k จาก scores จากโมเดล
            _, ind = scores.topk(k, 1, True, True)
                _ คือ ตัวแปรตัวยึดตำแหน่งที่เราไม่ต้องการ
                ind คือ tensor ของ indices ที่สอดคล้องกับค่า top k ใน scores
                k คือ จำนวนองค์ประกอบสูงสุดที่จะดึงข้อมูล
                1 คือ มิติข้อมูลเพื่อค้นหาองค์ประกอบ top k กรณีนี้ เป็นมิติที่สอง
                True คือ กำหนดว่าจะให้ส่งคืนค่า top k พร้อมกับ indices หรือไม่ กรณีนี้ ตั้งค่านี้เป็น True จะส่งคืนค่าทั้ง 2
                True คือ กำหนดว่าจะให้เรียงลำดับค่า top k จากมากไปหาน้อยหรือไม่ กรณีนี้ ตั้งค่านี้เป็น True จะเรียงลำดับค่า top k จากมากไปน้อย เพื่อให้องค์ประกอบแรกใน output มีค่ามากที่สุด
    """

    correct = ind.eq(targets.view(-1, 1).expand_as(ind)) # เปรียบเทียบ ind tensor กับ targets tensor (ซึ่งมี true label) หลังจากปรับรูปร่างของ targets tensor ใหม่ให้มีจำนวนคอลัมน์เท่ากันกับ ind tensor

    correct_total = correct.view(-1).float().sum()  # 0D tensor
                                                    # ปรับ correct tensor ให้เป็น tensor 1 มิติ จากนั้นคำนวณผลรวมขององค์ประกอบ

    return correct_total.item() * (100.0 / batch_size) # คำนวณ accuracy โดยการ นำจำนวนรวมของการทำนายที่ถูกต้อง หารด้วย batch size และคูณด้วย 100.0 เพื่อแปลงเป็นเปอร์เซ็นต์
