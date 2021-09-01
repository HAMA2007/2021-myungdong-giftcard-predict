# 2021-myungdong-giftcard-predict
명동 우천사와 우현사의 과거 시세와 Facebook의 Prophet을 이용하여 단기 미래를 예측해봅니다.(University Project)

## Requirements
- sqlalchemy
- pandas
- fbprophet
- plotly
- pymysql
- pytz
- twisted
- scrapy
- beautifulsoup4
- requests
- APScheduler
- pyyaml
- python-dateutil

## Crawl
- DB에 woocheon과 woohyun이라는 테이블을 만들고, datetime, ltt50, ltt10, ssg50, ssg10, hdi50, hdi10, ggs10, hps10항목을 만듭니다.
- datetime은 datetime형태로 하고, 나머지는 float형태로 하시면 됩니다.
- 이후, term_project.env에 DB정보를 수정하고 실행하면 됩니다.
- Dockerfile이나 Windows Shellscript 내용은 Report.ipynb에 있습니다.

## Predict
- term_project.py의 내용을 _run()함수를 주석 해제 혹은 주석 처리 하면서 이용하시면 됩니다.

## Disclaimer
- 추가 데이터 없이 시세 데이터만으로 예측을 진행하여, 미래에 이와 같은 시세가 된다고 보장하는 프로그램이 아닙니다. 해당 프로그램을 사용함으로서 생기는 책임은 모두 이용자 본인에게 있습니다.