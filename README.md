# Word2Vec 기반의 Recommendation system

> 함께 등장하는 song, tag는 서로 관련이 있을텐데 어떻게 학습시키면 좋을까?
>
> Word2vec을 이용하여 song과 tag의 정보를 학습합니다.

item 각각의 embedding을 학습하고, 이를 더하여 playlist embedding을 생성합니다.

나와 가장 유사한 주변 playlist 몇개를 추출하여 태그, 노래를 추천합니다.



## 1 ) item2vec 학습

skip gram 모델을 선택하여 3번 이상 등장한 item만 이용하여 128차원으로 학습시킨다.

일반적인 word2vec과는 다르게 등장 순서 자체는 관련이 없다. 그러므로 window size는 최대 길이인 210을 이용한다.

negative sample은 5개만 추출한다.



## 2 ) playlist embedding 생성하기

학습된 song, tag embedding을 모두 더하여 playlist embedding을 생성한다.

ex ) `{"songs": ['밤편지','푸르던', '거짓말처럼'] , ['잔잔한','사랑']` 의 embedding은 `vec_밤편지 + ... + vec_사랑`



## 3 ) 최적의 top N 찾기

유사한 playlist를 몇 개 찾아서 추천할 것인가 역시 성능에 큰 영향을 끼친다.

실험 결과 노래는 주변 10개 playlist를 , 태그는 50개 playlist를 참고하는 것이 가장 높은 점수를 얻을 수 있었다.

```python
주변 15개 참고 : Music nDCG: 0.207054
주변 10개 참고 : Music nDCG: 0.208711 ★
주변  5개 참고 : Music nDCG: 0.204856
```

```python
주변 65개 참고 : Tag nDCG: 0.452978
주변 50개 참고 : Tag nDCG: 0.454124 ★
주변 40개 참고 : Tag nDCG: 0.449345
```



## 4 ) 추천 결과 생성하기

> w2v 모델로 추천이 불가능한 경우에 baseline 모델을 이용하여 추천합니다.



playlist embedding을 정의할 수 없는 경우(애초에 tag,song 정보가 없거나, 있더라도 학습되지 않은 item인 경우)가 존재한다.



#### 1.  `GenreExpPopular` 에 의해 생성된 결과를 추천한다.

(`GenreExpPopular` 는 baseline의 `GenreMostPopular`의 개선모델으로, softmax를 이용하여 각 장르의 점수를 결정하고 추천한다.)

```python
Music nDCG: 0.208711
Tag nDCG: 0.454124
Score: 0.245523
```

[참고] GEP 모델만의 성능

```python
Music nDCG: 0.0416103
Tag nDCG: 0.162134
Score: 0.0596888
```



#### 2. AE 기반 모델에 의해 생성된 결과를 추천한다.

baseline 결과를 이전에 학습해놓은 AutoEncoder 기반 모델의 추천 결과로 바꾸면 최종 score에서 0.03을 향상시킬 수 있습니다.

```python
Music nDCG: 0.211357
Tag nDCG: 0.458047
Score: 0.24836
```

[참고] AE 기반 모델만의 성능

```python
Music nDCG: 0.110012
Tag nDCG: 0.342718
Score: 0.144918
```