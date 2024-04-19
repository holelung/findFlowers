import streamlit as st
import torch
import time
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms as T
import math

empty_img = 'images/Floworld_App_Logo.jpg'

st.set_page_config(
    page_title="플라월드", # html의 title과 같은 속성
    page_icon="images/Floworld_App_Logo.jpg"  # title의 아이콘 지정
)
probability = 0

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


st.title("플라월드")
st.markdown("**꽃**을 하나씩 추가해서 도감을 채워보세요!")
st.markdown("""
<style>
    /* 프로그레스 바의 색상 변경 */
    .stProgress > div > div > div > div {
        background-color: #a976c7;
    }
</style>
""", unsafe_allow_html=True)
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
        probability = math.trunc(max_probability*1000)/10
        flower_info = {
            "진달래": " 진달래는 보통 분홍색으로 많이 피고, 사랑과 기쁨을 뜻하는 꽃이랍니다!",
            "초롱꽃": " 전세계에 오직 1종밖에 없는 희귀식물로 오직 우리나라에서만 피는 꽃이에요! 물이 많고 습도가 높은 곳에서 자라며 대부분 연한 자주색을 띈답니다! 성실을 뜻하는 꽃이에요!",
            "능소화": " 여름에 피는 연한 주황색 꽃으로 꽃이 한 번만 피고 지지 않고 계속 피고, 또 피고 하여 꽃이 피는 기간 동안 계속 아름다운 꽃을 볼 수 있어요 추위에 약해서 9월이 되면 생기를 잃어버리는 꽃이에요 그리움과 기다림을 뜻하는 꽃이랍니다.",
            "벚꽃": " 봄이 되면 벚나무에서 활짝 피는 꽃으로 분홍색, 하얀색 꽃잎이 아주 화려하고 아름다운 꽃이에요 일본을 상징하는 꽃으로 아름다운 정신, 정신적 사랑을 뜻하는 꽃이랍니다.",
            "수레국화": " 국화의 한 종류로 30-90cm 남짓한 키에 보라색, 파란색 꽃잎이 특징적인 꽃이에요 4월-9월에 걸쳐서 피며 행복감을 뜻하는 꽃이랍니다!",
            "개나리": " 봄에 피는 대한민국의 고유 식물로 전국에서 쉽게 볼 수 있어요 전통적으로 봄이 왔음을 알리는 꽃으로 우리에게 매우 친근한 꽃이랍니다 희망과 깊은 정을 뜻하는 꽃이에요",
            "연꽃": " 인도에서 태어난 꽃으로 줄기는 우리가 먹는 연근으로 이용된답니다 물에서 피는 꽃이지만 논이나 늪지대에서도 찾아볼 수 있어요 소원해진 사랑, 깨끗한 마음을 뜻하는 꽃이랍니다",
            "나팔꽃": " 아침 일찍 피었다가 낮에는 오므라들면서 시드는 신기한 꽃이에요 줄기가 덩굴지고 2m까지 감긴답니다 일편단심 사랑을 뜻하는 꽃이에요",
            "무궁화": " 한국을 상징하는 사실상 국화로 생명력이 매우 강해서 많이 척박한 황경에서도 적응하고 피어나는 강한 꽃이에요 영원히 피고 또 피어서 지지 않는 다는 뜻을 가지고 있어요",
            "장미": " 높이는 2-3m이며 5~6월에 빨강색, 보라색, 흰색 등 아름다운 색으로 피어나서 관상용으로 키우는 경우가 많아요! 장미는 빛을 매우 좋아하고, 공기가 맛있고 영양분이 많은 땅에서 피어난답니다! 보통 낭만적인 사랑을 뜻하는 꽃이랍니다",
            "해바라기": " 콜럼버스가 아메리카를 발견한 이후 유럽에 소개되며 태양의 꽃으로 불리게 됐어요 줄기가 태양을 향해 굽어지는 특징이 있어서 해바라기라고 불리게 되었어요! 영원한 사랑을 뜻하는 꽃이에요"
        }
        if max_label in flower_info and probability>90:
            result = f"이 꽃은 {probability}% 확률로 **{max_label}** 입니다!"  # 꽃 이름을 강조체로 표시
            result += flower_info[max_label]
        else:
            result = "이 사진은 꽃이 아니거나, 등록된 꽃이 아닙니다."
            flower_name = "등록할 수 없는 꽃입니다."
    return flower_name, result, probability




if uploaded_image is not None:
    # 업로드 된 이미지 보여주기
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # 이미지를 텐서로 변환하고 모델에 적용하여 예측
    with st.spinner('Predicting...'):
        flower_name, prediction, probability = predict(image)

    
    st.session_state['name'] = flower_name
    if probability > 90 :
        st.write(prediction)
    else:
        st.error(prediction)
    # 예측 결과 출력


if "flowers" not in st.session_state:
    st.session_state.flowers = [
    {
        "name": "진달래",
        "image_url": empty_img
    },
    {
        "name": "초롱꽃",
        "image_url": empty_img,
    },
    {
        "name": "능소화",
        "image_url": empty_img,
    },
    {
        "name": "벚꽃",
        "image_url": empty_img
    },
    {
        "name": "수레국화",
        "image_url": empty_img
    },
    {
        "name": "개나리",
        "image_url": empty_img
    },
    {
        "name": "연꽃",
        "image_url": empty_img
    },
    {
        "name": "나팔꽃",
        "image_url": empty_img
    },
    {
        "name": "무궁화",
        "image_url": empty_img
    },
    {
        "name": "장미",
        "image_url": empty_img
    },
    {
        "name": "해바라기",
        "image_url": empty_img
    },
]
    

# 꽃 이름만 추출
flower_names = [flower["name"] for flower in st.session_state.flowers]
if probability == 0:
    enabled = False
    isPicture = False
    help_text = "사진을 업로드 해주세요"
elif probability >90:
    enabled = True
    isPicture = True
    help_text = ""
else:
    enabled = False
    isPicture = True
    help_text = "도감에 넣을 수 없는 사진입니다."

progress_text.text(f"{int(registered_images / 11 * 100)}% 완료")

if "registered_images" not in st.session_state:
    st.session_state.registered_images = 0

with st.form(key="form"):
    col1, col2 = st.columns(2)
    with col1:
        name=st.text_input(label="꽃 이름", value=flower_name if probability > 90 else "")

    image_url = uploaded_image
    submit = st.form_submit_button(label="Submit", help=help_text)
    if submit:
        if not enabled and not isPicture:
            st.error("사진을 업로드 해주세요.")
        elif not enabled and isPicture:
            st.error("도감에 등록할 수 있는 사진이 아닙니다.")
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


# CSS 스타일을 정의하여 이미지의 높이를 200px로 설정
st.markdown("""
<style>
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    img {
        width: 150px;  /* 너비는 150px로 고정 */
        height: 200px; /* 높이도 150px로 고정 */
    }
    [data-testid="StyledFullScreenButton"]{
        visibility:hidden
    }
""", unsafe_allow_html=True)

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
                        st.session_state.flowers[i+j]["image_url"] = "images/emty.png"
                        st.rerun()


