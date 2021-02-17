# Word2Vec 기반의 Recommendation system

> 함께 등장하는 song, tag는 서로 관련이 있을텐데, 어떻게 학습시키면 좋을까?
>
> Word2vec을 이용하여 song과 tag를 한 문장에 등장하는 단어로 보자 !
</br>

song과 tag를 단어로, playlist를 문장으로 보고 word2vec을 학습시킵니다.

학습된 item embedding을 모두 더해 playlist embedding을 생성하고,

나와 가장 유사한 주변 playlist k개를 추출하여 노래와 태그를 추천합니다.

```python
# 2021.02.10 현재 성능
Music nDCG: 0.220859
Tag nDCG: 0.512648
```
</br>

## 0 ) Data 전처리
train set의 tag를 vocabulary로 하여, title에서 tag를 추출한다.  

만약 vocabulary에 `Christmas, Jazz`가 있고, 입력된 title이 `christmas에 어울리는 jazz`라면, vocab에 있는 대소문자 형태와 맞추어 [`Christmas`, `Jass`]를 return하도록 한다.  

자세한 구현 방법 및 고민했던 부분들은 [haeuuu/Title-Based-Playlist-Generator/How to extract tag from title.md](https://github.com/haeuuu/Title-Based-Playlist-Generator/blob/master/How%20to%20extract%20tag%20from%20title.md)에서 확인할 수 있다.

</br>

## 1 ) word2vec 학습

skip gram 모델을 선택하여 3번 이상 등장한 item만 이용하여 128차원으로 학습시킨다.

일반적인 word2vec과는 다르게 등장 순서 자체는 관련이 없다. 그러므로 window size는 최대 길이인 210을 이용한다.

negative sample은 5개만 추출한다.

</br>

## 2 ) playlist embedding 생성하기

학습된 song, tag embedding을 모두 더하여 playlist embedding을 생성한다.

ex ) `{"songs": ['밤편지','푸르던', '거짓말처럼'] , ['잔잔한','사랑']` 의 embedding은 `vec_밤편지 + ... + vec_사랑`

</br>

### :thinking: 중요한 태그와 그렇지 않은 태그를 알려줄 방법은 없을까?

playlist의 제목이 `여름에 듣기 좋은 힙합` 이라고 하자.

 `힙합, 여름` 에 비하면  `노래, 듣기 좋은`과 같은 tag는 playlist를 대표한다고 말할 수 없다.

모두 같은 가중치로 이 태그들을 더하는 것보다, `힙합`과 `여름`의 힘은 더 **강하게**, `노래`와 `듣기 좋은`의 힘은 더 **약하게** 만들 방법이 필요하다.

그래서 tag가 얼마나 일관성을 갖는가에 대한 지표인 `consistency`라는 지표를 만들어보았다.

[haeuuu/Title-Based-Playlist-Generator/사용자의 의도 찾기.md](https://github.com/haeuuu/Title-Based-Playlist-Generator/blob/master/%EC%82%AC%EC%9A%A9%EC%9E%90%EC%9D%98%20%EC%9D%98%EB%8F%84%20%EC%B0%BE%EA%B8%B0.md) 에 더 자세한 설명을 써놓았다.

</br>

## 3 ) 최적의 top N 찾기

유사한 playlist를 몇 개 찾아서 추천할 것인가 역시 성능에 큰 영향을 끼친다.

실험 결과 노래는 주변 10개 playlist를 , 태그는 50개 playlist를 참고하는 것이 가장 높은 점수를 얻을 수 있었다!
