# 생체인증보안(24-1) 과제 보고서
먼저 테스트셋을 list_test2로 불러오고, list_test2[0]와 해당 이름과 일치하는 트레이닝 이미지 경로의 이미지를 각각 img1, img2로 저장했다. 모든 테스트셋과 트레이닝셋을 비교하기 전에, 이 두 이미지로 distance가 어느 정도 나오는지를 보고자 했다. 

## 원본 이미지

img1, img2를 각각 출력해보면 다음과 같이 나온다.
![1](https://github.com/user-attachments/assets/b2b96595-ed7a-4f4f-9ca5-f686fe5e1283)

## 이미지 전처리

### Pose - Affine Transform

img1에 대해 affine matrix 변환 과정을 출력해보고, 변환 전후의 좌표를 파란색과 빨간색으로 찍어보았다.
![2](https://github.com/user-attachments/assets/090bb7f9-9bc3-44fc-8a60-b7f975505c06)
![3](https://github.com/user-attachments/assets/5aa8fbc4-d1e5-461c-a454-01f45b6f5871)

45도 회전 결과는 다음과 같았다.
![4](https://github.com/user-attachments/assets/904f212d-302d-4052-a7ab-ee74bcbab201)


### Filtering

필터링이 굉장히 어려웠다. 직접 3x3 커널을 수정해보며 다음의 필터링이 그나마 결과가 괜찮아서 적용했다.

```python
ks = 3
kernel = np.array(( 
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
))
```

img1의 결과는 다음과 같았다.
![5](https://github.com/user-attachments/assets/4845e467-9cb9-4367-8507-0be4465587eb)


여전히 주변에 노이즈가 많고 융선이 또렷하지 않은 것 같았다. 그래서 가우시안 블러를 통해 노이즈를 감소하고, CLAHE를 이용해 명암을 균일화한 후, Otsu’s 이진화를 통해 다양한 필터링을 적용했다. 마지막에는 명암 반전을 했다.

```python
def process_image(image):
    # Step 1: 이미지를 그레이스케일로 변환 (이미 그레이스케일이면 필요 없음)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: 가우시안 블러를 적용하여 노이즈 감소
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Step 3: CLAHE를 적용하여 명암 균일화
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(blurred)
    
    # Step 4: Otsu의 이진화를 적용
    _, binary_image = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 5: 검정과 흰색 반전
    inverted_image = cv2.bitwise_not(binary_image)

    # 각 단계의 결과를 저장
    images = {
        'Original Image': image,
        'Gaussian Blurred Image': blurred,
        'Clahe Image' : clahe_image,\
        'Otsu\'s Binarized Image': binary_image,
        'Inverted Image': inverted_image
    }

    return images
```

이 과정을 이미지로 출력해보니 다음의 결과가 나왔다.
![6](https://github.com/user-attachments/assets/c94004ba-c0f8-4126-8132-1dbc2e591710)


여전히 마음에 들지 않는 부분은 융선들이 중간에 불필요하게 연결된 부분들과 주변의 노이즈이다. 최대한 선을 깔끔하게 만들어 보려고 Morphology 연산(침식, Opening) 등 다양한 필터링을 적용해보았지만, 저 결과가 최선이었다.

그래서 img2도 마찬가지로 직접 설정한 커널 이후 가우시안 필터링, Clahe, Otsu’s 이진화를 거쳐 다음의 결과를 얻었다.

![7](https://github.com/user-attachments/assets/bcace7d0-e733-4b6c-9aed-b2067db6e81b)

![8](https://github.com/user-attachments/assets/68a8d124-c656-4daf-b355-cc1b61792c83)

참고로 이진화되어있는지 확인하기 위해 다음과 같이 출력해보았다.

```python
fig, axes = plt.subplots(1,2,figsize = (18,5))
axes[0].hist(img1.ravel(), bins=256, color ="r");
axes[1].hist(img2.ravel(), bins=256);
```

결과는 이미지들이 이미 상당히 이진화되어 있음을 보였다.

![9](https://github.com/user-attachments/assets/db3efa2a-4179-4358-8f81-fce365428cb4)

## 특징 추출

특징 추출을 위해 `MinutiaeFeature` 클래스, `features_to_array` 함수, `getTerminationBifurcation` 함수, `computeAngle` 함수, `extractMinutiaeFeatures` 함수, `ShowResults` 함수를 작성했다.

`MinutiaeFeature` 클래스는 미세 특징을 저장하고 배열로 변환하며, `features_to_array` 함수는 여러 특징을 배열로 변환하고 패딩한다. `getTerminationBifurcation` 함수는 지문 이미지에서 종료점과 분기점을 추출하고, `computeAngle` 함수는 블록에서 미세 특징의 각도를 계산한다. `extractMinutiaeFeatures` 함수는 골격화된 이미지와 추출된 종료점 및 분기점에서 미세 특징을 추출하며, `ShowResults` 함수는 추출된 종료점과 분기점을 시각화한다.

이 함수들을 이용해 img1, img2에 적용해 스켈레톤화와 특징 추출을 진행한 결과, 각각 다음과 같이 나왔다.

![10](https://github.com/user-attachments/assets/13ad503a-d898-4c3f-ba03-32ddaaa545c1)

![11](https://github.com/user-attachments/assets/90007ba6-7be2-4875-92e7-f0920253d0b5)

그 후 특징점을 하나의 배열로 병합한 후 feat_query, feat_train와 각각의 갯수를 출력해보았다. 결과는 다음과 같았다.

```python
feat_query: [[  10.   52. -135.    0.    0.]
 [  10.   71. -180.    0.    0.]
 [  10.   81. -135.    0.    0.]
 ...
 [ 239.  122.   90. -180.  -45.]
 [ 239.   65.  135.   45. -135.]
 [ 240.   90.   90. -180.   -0.]]
feat_train: [[  10.   72. -180.    0.    0.]
 [  10.  102. -180.    0.    0.]
 [  10.  120. -180.    0.    0.]
 ...
 [ 239.  122.   90. -180.  -45.]
 [ 239.   65.  135.   45. -135.]
 [ 240.   90.   90. -180.   -0.]]
398 356
```

## 매칭

먼저 매칭하는 함수를 다음과 같이 짰다.

```python
def match_finger(feat_query, feat_train, threshold, use_orientation, img_query=None, img_train=None):
    matches = []
    for i, q_feat in enumerate(feat_query):
        q_loc = q_feat[:2]
        q_orientation = q_feat[2:]
        for j, t_feat in enumerate(feat_train):
            t_loc = t_feat[:2]
            t_orientation = t_feat[2:]

            # Compute Euclidean distance between query and train locations
            loc_dist = distance.euclidean(q_loc, t_loc)

            if use_orientation:
                # Compute orientation distance if required
                if np.isnan(q_orientation).any() or np.isnan(t_orientation).any():
                    ori_dist = float('inf')
                else:
                    ori_dist = distance.euclidean(q_orientation, t_orientation)
                total_dist = loc_dist + ori_dist
            else:
                total_dist = loc_dist

            if total_dist < threshold:
                matches.append((i, j, total_dist))

    # Filter matches to find the best ones
    matches = sorted(matches, key=lambda x: x[2])
    unique_matches_query = set()
    unique_matches_train = set()
    best_matches = []
    for match in matches:
        if match[0] not in unique_matches_query and match[1] not in unique_matches_train:
            best_matches.append(match)
            unique_matches_query.add(match[0])
            unique_matches_train.add(match[1])

    dist = sum([match[2] for match in best_matches])
    dist_half = dist / 2
    len_match = len(best_matches)

    # Optionally, visualize the matches if images are provided
    if img_query is not None and img_train is not None:
        combined_img = np.concatenate((img_query, img_train), axis=1)
        plt.figure(figsize=(10, 5))
        plt.imshow(combined_img, cmap='gray')
        cols = img_query.shape[1]
        
        for match in best_matches:
            q_loc = feat_query[match[0]][:2]
            t_loc = feat_train[match[1]][:2]
            t_loc[1] += cols  # Adjust t_loc for the combined image
            
            color = get_random_color()
            plt.plot([q_loc[1], t_loc[1]], [q_loc[0], t_loc[0]], '-', color=color)
            plt.plot(q_loc[1], q_loc[0], 'o', color=color)
            plt.plot(t_loc[1], t_loc[0], 'o', color=color)
        plt.show()

    return dist, dist_half, len_match

def get_random_color():
    return [random.random(), random.random(), random.random()]
```

이 함수를 이용해 dist, dist_half, len_match를 출력해보니 img1, img2의 매칭 결과는 다음과 같았다.

```python
dist, dist_half, len_match = match_finger(feat_query, feat_train, 50, True, img_query=img1, img_train=img2)
print(dist, dist_half, len_match)
```

![12](https://github.com/user-attachments/assets/5534a6e7-ae2e-4ec7-8f96-82514f941203)

dist는 대략 492.901, len_match는 336이었다.

dist가 어느 정도 되는 게 매칭되는 게 맞는지 확인하기 위해 두 번째 훈련 이미지도 로드하여 img2와 비교해보니 다음의 결과가 나왔다.

![13](https://github.com/user-attachments/assets/6e716f5c-ce7d-4fc3-98ca-dcdc98278581)

![14](https://github.com/user-attachments/assets/c14553b3-ea90-4e5b-a13d-36d3075bac40)

 dist는 대략 5375.110, len_match는 221이었다.

이 두 결과를 통해, 서로 동일한 지문의 경우 dist가 대략 500~550 이하이고, 다른 지문의 경우 dist가 4000~5000 이상이라고 판단하였다. 그래서 dist가 550 이하면 서로 일치하는 지문이라 판단하고 결과에 저장하도록 코드를 짰다.

위와 같이 이미지 세 개에 대해 분석을 한 결과를 바탕으로, 전체 훈련 이미지와 테스트 이미지에 대해 매칭을 수행하는 코드를 짜 실행해보았다. 매칭된 두 이미지와 distance를 출력하도록 했다.

![15](https://github.com/user-attachments/assets/11949472-e69f-46aa-8d01-a5a9a79280c4)

마지막 출력은 이런 결과를 나타냈다.

![16](https://github.com/user-attachments/assets/bc4b6bc7-1586-44bf-8f4a-5d028207b50c)

## Metric 계산

Precision, Recall, FAR, FRR, ACC를 계산하는 함수를 다음과 같이 짰다.

```python
def calculate_metrics(dict_result, threshold):
    TP = FP = FN = TN = 0
    for (test_img, train_img), (dist, dist_half, len_match, len_feat_query, len_feat_train) in dict_result.items():
        if dist < threshold:
            if os.path.basename(test_img).split('_')[0] == os.path.basename(train_img).split('_')[0]:
                TP += 1
            else:
                FP += 1
        else:
            if os.path.basename(test_img).split('_')[0] == os.path.basename(train_img).split('_')[0]:
                FN += 1
            else:
                TN += 1

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FRR = FN / (TP + FN) if (TP + FN) > 0 else 0
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return Precision, Recall, FAR, FRR, ACC
```

이제 임계값을 500으로 설정한 후 계산했더니 다음의 결과가 나왔다.

```python
Precision: 0.0, Recall: 0, FAR: 0.9217758985200846, FRR: 0, ACC: 0.07822410147991543
```
