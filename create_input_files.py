from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k', # dataset: ชื่อของ dataset ที่ใช้ ซึ่งในกรณีนี้คือ coco
                       karpathy_json_path='input_folder/captions/dataset_flickr8k.json', # karpathy_json_path: เส้นทางไปยังไฟล์ Karpathy JSON ที่มีรูปภาพและข้อมูลคำบรรยายสำหรับชุดข้อมูล
                       image_folder='input_folder/images/flickr8k_Images', # image_folder: เส้นทางไปยังโฟลเดอร์ที่มีรูปภาพของชุดข้อมูล
                       captions_per_image=5, # captions_per_image: จำนวนคำบรรยายที่จะสุ่มตัวอย่างสำหรับแต่ละภาพในชุดข้อมูล
                       min_word_freq=5, # min_word_freq: เกณฑ์ความถี่ขั้นต่ำสำหรับการรวมคำในคำศัพท์ที่ใช้สำหรับคำอธิบายภาพ
                       output_folder='output_folder/flickr8k_output', # output_folder: เส้นทางไปยังโฟลเดอร์ที่จะบันทึกไฟล์
                       max_len=50) # max_len: ความยาวสูงสุดของคำบรรยาย ในแง่ของจำนวนคำ