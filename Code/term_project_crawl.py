import datetime as dt
import pytz
import time

import argparse as ap
import os
from os import getenv

from typing import List
from types import SimpleNamespace as SimpleNs

from twisted.internet import reactor
from scrapy import Request, Spider
from scrapy.crawler import CrawlerProcess, CrawlerRunner
from scrapy.http import FormRequest
from scrapy.utils.log import configure_logging

from bs4 import BeautifulSoup
import requests

import sqlalchemy

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.twisted import TwistedScheduler

# --------------------------------------------------- #
# DB FUNCTIONS
# --------------------------------------------------- #
# CREATE DB ENGINE INTERNAL AND RETURN
def create_db_engine_with_pool(url:str = None):
    return sqlalchemy.create_engine(
        url,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,  # 30 seconds
        #pool_recycle=1800,  # 30 minutes
        pool_pre_ping=True,
    )

# CREATE DB ENGINE FRONT AND RETURN
def create_mysql_db_engine(host, port, username, password, database):
    return create_db_engine_with_pool(sqlalchemy.engine.url.URL(
        drivername='mysql+pymysql',
        username=username,
        password=password,
        database=database,
        host=host,
        port=port
    ))

# --------------------------------------------------- #
# LOGGER FUNCTIONS
# --------------------------------------------------- #
import logging
import logging.config
import yaml
from log_formatter import Basic

with open('config.yaml') as config_file:
    config = yaml.full_load(config_file).get('log', None)
    if config:
        logging.config.dictConfig(config)
logger = logging.getLogger('term_project_crawl')


# --------------------------------------------------- #
# CRAWLER FUNCTIONS (WOOHYUN WITH SCRAPY)
# --------------------------------------------------- #
class WOOHYUNCrawlSpider(Spider):
    name = 'woohyun_crawler'
    _WOOHYUN_URL = 'https://wooh.co.kr/'

    def __init__(self, db_eng, **kwargs):
        super().__init__(self.name, **kwargs)
        self.db_eng = db_eng

    def crawl_func(self, response):
        now_time = dt.datetime.now(tz=pytz.timezone('Asia/Seoul'))
        now_time_naive = now_time.replace(tzinfo=None)

        if response.status == 200:
            logger.info(f'(WOOHYUN) DATA PARSING...')
            # LOTTE 10
            ltt10_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[2]/td[2]/h/text()').get()
            ltt10 = (ltt10_raw.split('(')[1]).split('%)')[0]
            # LOTTE 50
            ltt50_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[1]/td[2]/h/text()').get()
            ltt50 = (ltt50_raw.split('(')[1]).split('%)')[0]
            # SHINSEGAE 10
            ssg10_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[7]/td[2]/h/text()').get()
            ssg10 = (ssg10_raw.split('(')[1]).split('%)')[0]
            # SHINSEGAE 50
            ssg50_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[6]/td[2]/h/text()').get()
            ssg50 = (ssg50_raw.split('(')[1]).split('%)')[0]
            # HYUNDAI 10
            hdi10_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[11]/td[2]/h/text()').get()
            hdi10 = (hdi10_raw.split('(')[1]).split('%)')[0]
            # HYUNDAI 50
            hdi50_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[12]/td[2]/h/text()').get()
            hdi50 = (hdi50_raw.split('(')[1]).split('%)')[0]
            # GUKMIN GWANGWANG 10
            ggs10_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[16]/td[2]/h/text()').get()
            ggs10 = (ggs10_raw.split('(')[1]).split('%)')[0]
            # HOMEPLUS 10
            hps10_raw = response.xpath('//*[@id="idx_hit"]/div[1]/table/tbody/tr[24]/td[2]/h/text()').get()
            hps10 = (hps10_raw.split('(')[1]).split('%)')[0]

            try:
                sql = '''INSERT IGNORE INTO woohyun VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
                with self.db_eng.connect() as dbconn:
                    dbconn.execute(sql, [now_time_naive, ltt50, ltt10, ssg50, ssg10, hdi50, hdi10, ggs10, hps10])
                logger.info(f'(WOOHYUN) DATA INSERT DONE')
            except:
                logger.error(f'(WOOHYUN) DATA INSERT ERROR')
        else:
            logger.error(f'(WOOHYUN) CRAWL FAILED')

    def start_requests(self):
        headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'}
        yield Request(self._WOOHYUN_URL, headers=headers, callback=self.crawl_func)

# --------------------------------------------------- #
# CRAWLER FUNCTIONS (WOOCHEON WITH BS4)
# --------------------------------------------------- #
def woocheon_crawl_bs4(db_eng):
    # Load Site Datas
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'}
    site_url = 'http://www.wooticket.com/popup_price.php'
    response = requests.get(site_url, headers=headers)
    soup = BeautifulSoup(response.content.decode('euc-kr', 'replace'), 'html.parser')
    
    now_time = dt.datetime.now(tz=pytz.timezone('Asia/Seoul'))
    now_time_naive = now_time.replace(tzinfo=None)

    if response.status_code == 200:
        logger.info(f'(WOOCHEON) DATA PARSING...')
        # Parse Common Part
        soup1 = soup.find('table')
        soup2 = soup1.find('table')
        table_list = soup2.find_all('table')[1]
        column_list = table_list.find_all('tr')

        # Parse Individual Part
        ltt50_line = column_list[2].find_all('div')
        ltt50 = (ltt50_line[1].text.split('('))[1].split('%)')[0]
        ltt10_line = column_list[3].find_all('div')
        ltt10 = (ltt10_line[1].text.split('('))[1].split('%)')[0]

        ssg50_line = column_list[7].find_all('div')
        ssg50 = (ssg50_line[1].text.split('('))[1].split('%)')[0]
        ssg10_line = column_list[8].find_all('div')
        ssg10 = (ssg10_line[1].text.split('('))[1].split('%)')[0]

        hdi50_line = column_list[11].find_all('div')
        hdi50 = (hdi50_line[1].text.split('('))[1].split('%)')[0]
        hdi10_line = column_list[12].find_all('div')
        hdi10 = (hdi10_line[1].text.split('('))[1].split('%)')[0]

        ggs10_line = column_list[14].find_all('div')
        ggs10 = (ggs10_line[1].text.split('('))[1].split('%)')[0]

        hps10_line = column_list[16].find_all('div')
        hps10 = (hps10_line[1].text.split('('))[1].split('%)')[0]
        
        crawled_data = SimpleNs(
                        now_time_naive=now_time_naive,
                        ltt50=ltt50,
                        ltt10=ltt10,
                        ssg50=ssg50,
                        ssg10=ssg10,
                        hdi50=hdi50,
                        hdi10=hdi10,
                        ggs10=ggs10,
                        hps10=hps10
        )
        _insert_to_woocheon_db(db_eng, crawled_data)
    else:
        logger.error(f'(WOOCHEON) CRAWL FAILED')
    

def _insert_to_woocheon_db(db_eng, crawled_data:SimpleNs):
    try:
        sql = '''INSERT INTO woocheon VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
        with db_eng.connect() as dbconn:
            dbconn.execute(sql, [crawled_data.now_time_naive, 
                                 crawled_data.ltt50, 
                                 crawled_data.ltt10, 
                                 crawled_data.ssg50, 
                                 crawled_data.ssg10, 
                                 crawled_data.hdi50, 
                                 crawled_data.hdi10, 
                                 crawled_data.ggs10, 
                                 crawled_data.hps10])

        logger.info(f'(WOOCHEON) DATA INSERT DONE')
    except:
        logger.error(f'(WOOCHEON) DATA INSERT ERROR')
    finally:
        return

# --------------------------------------------------- #
# WOOHYUN DUE TO APSCHEDULER FAIL
# --------------------------------------------------- #
def _crawl_woohyun(ctx):
    logger.info(f'(WOOHYUN) CRAWL STARTED')
    ctx.runner.crawl(WOOHYUNCrawlSpider, ctx.db_eng).addCallback(_sched_woohyun, ctx)
def _sched_woohyun(_, ctx):
    # interval is int and minute
    interval = ctx.interval

    now = dt.datetime.now(tz=pytz.timezone('Asia/Seoul'))
    
    nextworkdt = now + dt.timedelta(minutes=interval)
    nextworkdt = nextworkdt.replace(second=0, microsecond=0)
    change_minute = (int(int(int(nextworkdt.minute) / interval) * interval))
    nextworkdt = nextworkdt.replace(minute=change_minute)
    wait_seconds = (int((nextworkdt - now).total_seconds()) + 1)

    logger.info(f'(WOOHYUN) - NOW:{now}, NEXT:{nextworkdt}, WAIT:{wait_seconds}sec')

    reactor.callLater(wait_seconds, _crawl_woohyun, ctx)


# --------------------------------------------------- #
# SCHED ENTRY
# --------------------------------------------------- #
def _run_cron():
    logger.info(f'(INIT) CRAWL GIFTCARD MODULE STARTED')

    # READ FROM ENV
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='./term_project.env')

    # CREATE DB ENGINE
    db_eng = create_mysql_db_engine(
                host=getenv('DB_HOST', None),
                port=getenv('DB_PORT', None),
                username=getenv('DB_USER', None),
                password=getenv('DB_PASS', None),
                database=getenv('DB_NAME', None)
    )
    logger.info('(INIT) DB OK')

    # APScheduler
    executors = {
        'threadpool': ThreadPoolExecutor(2),
    }
    job_defaults = {'coalesce': False, 'max_instances': 2}
    sched = BackgroundScheduler(executors=executors, job_defaults=job_defaults)

    # CRON PERIOD
    cron_minute = getenv('CRON_MINUTE', 5)
    cron = f'*/{cron_minute}'
    logger.info(f'CRON PERIOD - {cron}minute')


    runner = CrawlerRunner(settings={
            'LOG_LEVEL': 'INFO', 
            'CONCURRENT_REQUESTS': 1
        })

    # WOOCHEON CRAWL
    sched.add_job(woocheon_crawl_bs4,
                  args=[db_eng],
                  executor='threadpool',
                  trigger='cron',
                  minute=cron,
                  id='crawl_woocheon')
    logger.info(f'(WOOCHEON) CRAWL SCHED START')
    sched.start()
    logger.info(f'(INIT) APSCHED OK')

    # WOOHYUN CRAWL
    ctx = SimpleNs(db_eng=db_eng, runner=runner, interval=cron_minute)
    logger.info(f'(WOOHYUN) CRAWL SCHED START')
    _crawl_woohyun(ctx)

    # runner.start(False)
    reactor.run()


    while True:
        time.sleep(10000)


# -------------------------------------------------- #
# TEST FUNCTION
# -------------------------------------------------- #    
def _run_direct():
    '''FOR TEST ONLY
    '''
    # FOR TEST, LOAD ENV DIRECT
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='./term_project.env')

    # CREATE DB ENGINE
    db_eng = create_mysql_db_engine(
                host=getenv('DB_HOST', None),
                port=getenv('DB_PORT', None),
                username=getenv('DB_USER', None),
                password=getenv('DB_PASS', None),
                database=getenv('DB_NAME', None)
    )

    # 1. CRAWL WOOHYUN WITH SCRAPY
    runner = CrawlerRunner(settings={
        'LOG_LEVEL': 'DEBUG', 'CONCURRENT_REQUESTS': 1
    })
    runner.crawl(WOOHYUNCrawlSpider, db_eng)

    # 2. CRAWL WOOCHEON WITH BS4
    woocheon_crawl_bs4(db_eng)

    # FOR SCRAPY
    reactor.run()

# -------------------------------------------------- #
# ENTRY POINT
# -------------------------------------------------- #    
def _run():
    _run_cron()
    # _run_direct()

if __name__ == '__main__':
    _run()