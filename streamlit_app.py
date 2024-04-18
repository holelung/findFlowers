import streamlit as st
import torch
import time
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms as T

st.set_page_config(
    page_title="꽃 도감", # html의 title과 같은 속성
    page_icon="images/logo.jpeg"  # title의 아이콘 지정
)

# 사전 학습된 ResNET 모델
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model_path = "models/pytorch_flowers_pretrained_resnet50_epoch_2.pth"

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

num_classes = 11

num_features = model.fc.out_features

# 분류층 정의
fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=num_classes),
)

# 사전 정의된 VGG 16 모델의 분류기를 변경
model.classifier = fc

model.eval()

data_transform = T.Compose([
    T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.3831408, 0.34173775, 0.28202718), std=(0.3272624, 0.29501674, 0.30364394)),
])

labels_file = "models/modelspytorch_flowers_labels.txt"
with open(labels_file, "r", encoding='UTF-8') as f:
    labels = [line.strip() for line in f.readlines()]


st.title("Flower Book")
st.markdown("**꽃**을 하나씩 추가해서 도감을 채워보세요!")
progress_bar = st.progress(0)
progress_text = st.empty()
registered_images = 0 # 등록한 이미지를 저장하는 변수 

uploaded_image = st.file_uploader("사진 찍은 이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])
def predict(image):
    # 이미지를 텐서로 변환
    inp = data_transform(image).unsqueeze(0)
    print("Input image tensor shape:", inp.shape)  # 입력 이미지 텐서의 모양 확인
    with torch.no_grad():
        output = model(inp)[0]
        print("Model output tensor:", output)  # 모델 출력 텐서 확인
        prediction = torch.nn.functional.softmax(output, dim=0)
        probabilities = {labels[i]: float(prediction[i]) for i in range(11)}
        max_label = max(probabilities, key=probabilities.get)
        flower_name = max_label
        max_probability = probabilities[max_label]
        result = "이 꽃은 {} 입니다!".format(max_label)
    return flower_name, result




if uploaded_image is not None:
    # 업로드 된 이미지 보여주기
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # 이미지를 텐서로 변환하고 모델에 적용하여 예측
    with st.spinner('Predicting...'):
        flower_name, prediction = predict(image)

    st.session_state['name'] = flower_name
    st.write(prediction)
    # 예측 결과 출력



type_emoji_dict = {
    "진달래": "🌸",
    "초롱꽃": "🌼",
    "능소화": "🌺",
    "벚꽃": "🌸",
    "수레국화": "🌸",
    "개나리": "🌼",
    "연꽃": "🌸",
    "나팔꽃": "🌸",
    "무궁화": "🌸",
    "장미": "🌹",
    "해바라기": "🌻",

}


if "flowers" not in st.session_state:
    st.session_state.flowers = [
    {
        "name": "진달래",
        "image_url": "images/emty.png"
    },
    {
        "name": "초롱꽃",
        "image_url": "images/emty.png",
    },
    {
        "name": "능소화",
        "image_url": "images/emty.png",
    },
    {
        "name": "벚꽃",
        "image_url": "images/emty.png"
    },
    {
        "name": "수레국화",
        "image_url": "images/emty.png"
    },
    {
        "name": "개나리",
        "image_url": "images/emty.png"
    },
    {
        "name": "연꽃",
        "image_url": "images/emty.png"
    },
    {
        "name": "나팔꽃",
        "image_url": "images/emty.png"
    },
    {
        "name": "무궁화",
        "image_url": "images/emty.png"
    },
    {
        "name": "장미",
        "image_url": "images/emty.png"
    },
    {
        "name": "해바라기",
        "image_url": "images/emty.png"
    },
]

# 꽃 이름만 추출
flower_names = [flower["name"] for flower in st.session_state.flowers]


progress_text.text(f"{int(registered_images / 11 * 100)}% 완료")

if "registered_images" not in st.session_state:
    st.session_state.registered_images = 0

with st.form(key="form"):
    col1, col2 = st.columns(2)
    with col1:
        name=st.text_input(label="꽃 이름", value=st.session_state.get('name', ''))

    image_url = uploaded_image
    submit = st.form_submit_button(label="Submit")
    if submit:
        if not name:
            st.error("꽃의 이름을 입력해주세요.")
        elif name in flower_names:
            updated = False
            for flower in st.session_state.flowers:
                if flower["name"] == name:
                    flower["image_url"] = uploaded_image.getvalue()
                    updated = True
                    st.success("이미지 업로드 완료")
                    st.session_state.registered_images += 1 
                    break 
        
            progress_bar.progress(st.session_state.registered_images / 11)
            progress_text.text(f"{int(st.session_state.registered_images / 11 * 100)}% 완료")
        else:
            st.error(f'{name}은 도감에 넣을 수 없는 사진입니다.')




for i in range(0, len(st.session_state.flowers), 3):
    row_flowers = st.session_state.flowers[i:i+3]
    cols = st.columns(3)
    for j in range(len(row_flowers)):
        with cols[j]:
            flower = row_flowers[j]
            with st.expander(label=f"**{i + j + 1}. {flower['name']}**", expanded=False if not flower['image_url'] else True):
                if flower['image_url']:
                    st.image(flower["image_url"], width=150, use_column_width=False)
                    delete_button = st.button(label="삭제", key=i+j, use_container_width=True)
                    if delete_button:
                        print("delete button clicked!")
                        del st.session_state.flowers[i+j]
                        st.rerun()


#css
st.markdown("""   
<style>
    img{
        max-width: 150px;  
        max-height: auto;  
    }
 
    .st-emotion-cache-1clstc5.eqpbllx1 {
        display:flex;
        justify-content : center;
    }
</style>
""", unsafe_allow_html=True)