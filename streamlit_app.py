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
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 180)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomVerticalFlip(p=0.5),
        T.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        T.ToTensor(),
        T.Normalize(mean=(0.3831408, 0.34173775, 0.28202718), std=(0.3272624, 0.29501674, 0.30364394)),
])

labels_file = "models/modelspytorch_flowers_labels.txt"
with open(labels_file, "r", encoding='UTF-8') as f:
    labels = [line.strip() for line in f.readlines()]

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
        st.session_state.name = max_label
        max_probability = probabilities[max_label]
        result = "이 꽃은 {} 입니다!".format(max_label)
    return result




if uploaded_image is not None:
    # 업로드 된 이미지 보여주기
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # 이미지를 텐서로 변환하고 모델에 적용하여 예측
    with st.spinner('Predicting...'):
        prediction = predict(image)

    st.write(prediction)
    # 예측 결과 출력



print("page reloaded")

st.title("Flower Book")
st.markdown("**꽃**을 하나씩 추가해서 도감을 채워보세요!")
progress_bar = st.progress(10)



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



initial_flowers = [
    {
        "name": "진달래",
        "types": ["진달래"],
        "image_url": "https://i.namu.wiki/i/gtPnuI2PUHQ-0oHqv6TtZ1TdGEkSCtmG6j6si7W8Rlf5pzl6cEQfDLEml-EkxgcqC0yxnQf6h-HbwFp3TWjLFUTbsoOAbBwaDGHN-0PyX2IgwNHOZTY4J914nama0tV6pFyIwYNJLSPCMH8B3mBlhA.webp"
    },
    {
        "name": "초롱꽃",
        "types": ["초롱꽃"],
        "image_url": "https://i.namu.wiki/i/H2CVRTzeyJea3cLRxS7ONxARUjh2GKnvSgN1QqZeGDxTYodfV6_NG7INz5jLWCLPCN0m6ysk7iaZwk44iTANWgzWr0Z7yJo2HAmO0pK5Oi4im4HwirGWeAKvlQLfugzFx0tuSdXem8GALtBD8IA63A.webp",
    },
    {
        "name": "능소화",
        "types": ["능소화"],
        "image_url": "https://i.namu.wiki/i/3UjWNPR2fMhZ-4ygfSI2WttZ-CbppJByXsp1CXojq_cI_0byg5fnX-CNrjAubVlUzRAsYeON78foWhAxlag3C9DKDE6O1GatOj9oQaQpRK8FuZaWK_BXNphS7MbwtN5gDSgDY6wNKZNgBllvqxaVGQ.webp",
    },
    {
        "name": "벚꽃",
        "types": ["벚꽃"],
        "image_url": "https://i.namu.wiki/i/m5gxmeZesC7mUhKi5mJzz8RjT35JUfUGmQSYLeCXi7ppDMh2j1lYNP68QwU5ha3B1M9nSzNxMG46XGuWvqqup8VGEm2ApscB3E3vM6yynUGF-lEWndxHdhLSTlRUpZoVwyR0TNObJg1X0Aqxn8MatA.webp"
    },
    {
        "name": "수레국화",
        "types": ["수레국화"],
        "image_url": "https://i.namu.wiki/i/iwar-zRfqOYgdoR-JGfLL8FWE8xcWOnX1HgT743winbvrTCqA8wkXM9WYz-kGRDqv_c619KL59rYF_5-Ln4IkUzdbDU38BSo1Dmz1X9QxUX3_Ty8F731QBk8AxaObPmFYS7SWWAAjuzG73NWSZPcBA.webp"
    },
    {
        "name": "개나리",
        "types": ["개나리"],
        "image_url": "https://i.namu.wiki/i/jsZgoEfGcacBEL-xF7fUJCLEup6uOqLlMbeQtqsONBmnNzPZxiVmzgBM-_YH7eRKMKSaERFkm5N8viLu80iC8ildkFrAo7xlUm5LtLAOuDX68ywQOvctMcCgRRQDs305-fPHciLg7GXXf1cxAn5Gcg.webp"
    },
    {
        "name": "연꽃",
        "types": ["연꽃"],
        "image_url": "https://i.namu.wiki/i/5fEek9xxvCWaXTax7AWZrW_O5B_JHDSvjbROXeHd_NTKweGc0j9halfuYCGCyYThBIevTdYKOq7p-rcf5qA5DVK65gH1J1bOiUwBSCYvcJyJNJ76eJwcs9dIeHofuv9n_YLbRUsU7eDPLeviT3wrZg.webp"
    },
    {
        "name": "나팔꽃",
        "types": ["나팔꽃"],
        "image_url": "https://i.namu.wiki/i/ZvlXBrNewlN6gqEg65YeTt5kj6W9ExM1EAHiHloLJqGPjacjtsUO7A3q2whWAgj88DCMSHo0uwNDl8h4cZPt_c8nIrq_cHS16U-QTFT0mnsHJ4rz6CNt98DIxzkzzrxmgLySdOlgugh0wKX-iuqfnA.webp"
    },
    {
        "name": "무궁화",
        "types": ["무궁화"],
        "image_url": "https://i.namu.wiki/i/et06cFLsf8lgtBrCSmXRe5BVJ3iAii2XUfWqHAXW18GBXkktejNkWIuCa4vioF2ydnJMc2Y4XT44L8HNkO5grNsHoBSvgzHpLe9sTxrd6vpGEX4PPmFuCm_aduXT8drGeF_LkBHbAPt8wSqDfH5jdg.webp"
    },
    {
        "name": "장미",
        "types": ["장미"],
        "image_url": "https://i.namu.wiki/i/N77ZYIeJO038FBOOQYU5NtW4ZZWyiMxIIf4ULpmGjb8s7DU4PzbZD8WzOzFJczPplff2LWC1URdmwqDTiE1Da_t-NbJCZXV9Gs2-IJk993chK1vTpWHFBmbu0UB7IR82Lyp1H0LArtCHFfnQQnxeFw.webp"
    },
    {
        "name": "해바라기",
        "types": ["해바라기"],
        "image_url": "https://i.namu.wiki/i/MJS06mJRUQejrkOL4lYHz2qUdN1Hf7f9BzuAiDMXkSjIHzImCD-lN8EqCXaMNOoJdL-7OyO4QCXUsGDAqzHoKMW1xa8A_T5ISQSGrsu5fQ3yU-MfbFsaca-jqaAOoAzD2hIgMM9uh9b5jq8qXozF4g.webp"
    },
]

example_flower = {
    "name" : "장미",
    "types" : "장미",
    "image_url" : "https://i.namu.wiki/i/N77ZYIeJO038FBOOQYU5NtW4ZZWyiMxIIf4ULpmGjb8s7DU4PzbZD8WzOzFJczPplff2LWC1URdmwqDTiE1Da_t-NbJCZXV9Gs2-IJk993chK1vTpWHFBmbu0UB7IR82Lyp1H0LArtCHFfnQQnxeFw.webp"
}

if "flowers" not in st.session_state:
    st.session_state.flowers = initial_flowers


auto_complete = st.toggle("예시 데이터로 채우기")
with st.form(key="form"):
    col1, col2 = st.columns(2)
    with col1:
<<<<<<< HEAD
        name=st.text_input(
            label="꽃 이름",
            value=example_flower["name"] if auto_complete else ""
        )
=======
        name=st.text_input(label="꽃 이름", value=st.session_state.get('name', ''))
>>>>>>> 9bec33125aeb7a83bb829eb2b20cd3580cdc89ce

    with col2:

        types = st.multiselect(label = "꽃 속성", options = list(type_emoji_dict.keys()))
    image_url = st.text_input(label="꽃 이미지 URL")
    submit = st.form_submit_button(label="Submit")
    if submit:
        if not name:
            st.error("꽃의 이름을 입력해주세요.")
        elif len(types) ==0:
            st.error("꽃의 속성을 적어도 한 개 선택해주세요.")
        else:
            st.success("포켓몬을 추가할 수 있습니다.")
            st.session_state.flowers.append({
                "name": name,
                "types": types,
                "image_url": image_url if image_url else "./images/Rose_1.jpg"

            })




for i in range(0, len(st.session_state.flowers), 3):
    row_flowers = st.session_state.flowers[i:i+3]
    cols = st.columns(3)
    for j in range(len(row_flowers)):
        with cols[j]:
            flower = row_flowers[j]
            with st.expander(label=f"**{i + j + 1}. {flower['name']}**",expanded=True):  # 아래 화살표 누르면 나오게 /expanded : 페이지 열면 펼쳐져 있게
                st.image(flower["image_url"])
                emoji_types = " ".join([f"{type_emoji_dict[x]} {x}" for x in flower["types"]])
                st.subheader(emoji_types)
                delete_button = st.button(label="삭제", key=i+j, use_container_width=True)
                if delete_button:
                    print("delete button clicked!")
                    del st.session_state.flowers[i+j]
                    st.rerun()


#css
st.markdown("""   
<style>
    img{
        max-height:300px;
    }
 
    [data-testid="StyledFullScreenButton"] {
        visibility : hidden
    }
</style>
""", unsafe_allow_html=True)