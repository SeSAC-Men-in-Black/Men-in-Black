## 진행 과정:

1. 차량 인식
2. 차선 인식
3. 위반 탐지지

## 모델 구성 및 분류:

1. 차량 인식 모델
   
    a. 모델 구성
   
       ⅰ. Detection Model : Mask R-CNN
   
       ⅱ. BackBone Network : ResNet101
   
       ⅲ. BackBone Pre-trained : torchvision://resnet101
   
       ⅳ. Loss function : SeesawLoss
   
       ⅴ. Optimizer : SGD, lr 초기값: 1e-6
   
    b. Class 분류
   
       ⅰ. 이륜차(vehicle_bike) : 10066
   
       ⅱ. 버스(vehicle_bus) : 75198
   
       ⅲ. 승용차(vehicle_car) : 232013
   
       ⅳ. 트럭(vehicle_truck) : 28905

2. 차선 인식 모델

    a. 모델 구성
   
       ⅰ. Detection Model : FCN(Fully Convolutional Network)
   
       ⅱ. BackBone Network : ResNet50
   
       ⅲ. Loss function : FocalLoss
   
       ⅳ. Optimizer : Adam, lr 초기값: 0.001
   
    b. Class 분류
   
       ⅰ. 색상별
   
           1) 청색(lane_blue) : 133654
   
           2) 갓길차선(lane_shoulder) : 55639
   
           3) 흰색(lane_white) : 128181
   
           4) 황색(lane_yellow) : 29554
   
       ⅱ. 타입별
   
            1) 1줄 점선(single_dashed) : 78953
   
            2) 1줄 실선(single_solid) : 181342
   
            3) 2줄 실선(double_solid) : 84914
   
            4) 좌점선_우실선(left_dashed_double) : 1095
   
            5) 좌실선_우점선(right_dashed_double) : 724
   

3. 위반 탐지 모델

    a. 모델 구성
   
       ⅰ. Detection Model : ResNet18
   
       ⅱ. Loss function : CrossEntropyLoss
   
       ⅲ. Optimizer : SGD, lr 초기값: 0.001
   
    b. Class 분류
   
       ⅰ. 정상(normal): 197618
   
       ⅱ. 위험(danger): 31229
   
       ⅲ. 위반(violation): 117335
   
    c. 위반 탐지 과정
   
       ⅰ. 정상
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/2e074200-ff13-47c8-9781-ea10440611ae)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/a0ce2c45-06a2-4d8b-a9a9-1856df83fd89)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/8fc3ccee-c625-48f9-a12c-6e77046a8507)
   
       ⅱ. 위험
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/7d8d9cea-9e1d-4f7b-8391-93abd1474d1b)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/f463b8a0-07ad-4f24-b5fb-65cd146aa369)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/4e0f5535-8d59-4045-87c9-d13ea7040ba2)
   
        ⅲ. 위반
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/139735ea-e164-4e1f-9834-fd0bec7cd076)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/8cc27f2e-a4d5-484e-b559-e01796cd88c3)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/1c059bb2-0456-4fd9-bcd8-7a576c1e315c)



