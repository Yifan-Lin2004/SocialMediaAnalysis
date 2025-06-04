import os
from bs4 import BeautifulSoup
import pandas as pd
import json
from urllib import parse
import requests
import time
import random

# 设置为自己的cookies（为隐私考虑就不在此展示了）
cookies = {
    "SINAGLOBAL": "",
    "ALF": "",
    "SUB": "",
    "SUBP": "",
    "_s_tentry": "",
    "Apache": "",
    "ULV": "",
    "WBPSESS": ""
}


def get_the_list_response(q='话题', the_type='实时', p='页码', timescope="2024-03-01-0:2024-03-27-16"):
    """
    q表示的是话题，type表示的是类别，有：综合，实时，热门，高级，p表示的页码，timescope表示高级的时间，不用高级无需带入
    """
    type_params_url = {
        '综合': [{"q": q, "Refer": "weibo_weibo", "page": p, }, 'https://s.weibo.com/weibo'],
        '实时': [{"q": q, "rd": "realtime", "tw": "realtime", "Refer": "realtime_realtime", "page": p, },
                 'https://s.weibo.com/realtime'],
        '热门': [{"q": q, "xsort": "hot", "suball": "1", "tw": "hotweibo", "Refer": "realtime_hot", "page": p},
                 'https://s.weibo.com/hot'],
        # 高级中的xsort删除后就是普通的排序
        '高级': [{"q": q, "xsort": "hot", "suball": "1", "timescope": f"custom:{timescope}", "Refer": "g", "page": p},
                 'https://s.weibo.com/weibo']
    }

    params, url = type_params_url[the_type]

    headers = {
        'authority': 's.weibo.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'referer': url + "?" + parse.urlencode(params).replace(f'&page={params["page"]}',
                                                               f'&page={int(params["page"]) - 1}'),
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69',
    }
    response = requests.get(url, params=params, cookies=cookies, headers=headers)
    return response


def parse_the_list(text):
    """该函数就是解析网页主题内容的"""
    soup = BeautifulSoup(text)
    divs = soup.select('div[action-type="feed_list_item"]')
    lst = []
    for div in divs:
        mid = div.get('mid')
        uid = div.select_one('div.card-feed > div.avator > a')
        if uid:
            uid = uid.get('href').replace('.com/', '?').split('?')[1]
        else:
            uid = None
        time = div.select_one('div.card-feed > div.content > div.from > a:first-of-type')
        if time:
            time = time.string.strip()
        else:
            time = None
        p = div.select_one('div.card-feed > div.content > p:last-of-type')
        if p:
            p = p.strings
            content = '\n'.join([para.replace('\u200b', '').strip() for para in list(p)]).replace('收起\nd', '').strip()
        else:
            content = None
        
        # 不再提取转发、评论和点赞数
        retweets = None
        comments = None
        star = None

        lst.append((mid, uid, content, time))
    # 更新列名，移除转发、评论、点赞
    df = pd.DataFrame(lst, columns=['mid', 'uid', 'content', 'time'])
    return df


def get_the_list(q, the_type, max_pages=50):
    """
    爬取指定话题下的所有微博内容
    :param q: 话题关键词
    :param the_type: 搜索类型（综合/实时/热门/高级）
    :param max_pages: 最大爬取页数，默认50页
    :return: 包含所有微博数据的DataFrame
    """
    df_list = []
    page = 1
    
    while page <= max_pages:
        try:
            response = get_the_list_response(q=q, the_type=the_type, p=page)
            if response.status_code == 200:
                df = parse_the_list(response.text)
                if df.empty:  # 如果返回的数据为空，说明已经到达最后一页
                    break
                df_list.append(df)
                print(f'第{page}页解析成功！', flush=True)
                
                # 添加随机延时，避免被封
                time.sleep(random.uniform(1, 3))
                page += 1
            else:
                print(f'第{page}页请求失败，状态码：{response.status_code}')
                break
        except Exception as e:
            print(f'爬取第{page}页时发生错误：{str(e)}')
            break
    
    if df_list:
        final_df = pd.concat(df_list)
        return final_df
    return pd.DataFrame()


if __name__ == '__main__':
    # 设置爬取参数
    the_type = '实时'
    q = '#年轻人恐婚恐育的真正原因是什么#'
    
    # 爬取数据
    df = get_the_list(q, the_type)
    
    # 保存数据
    if not df.empty:
        # 确保data目录存在
        os.makedirs('data', exist_ok=True)
        filename = f'data/{q.replace("#", "").replace(" ", "_")}.csv'
        try:
            df.to_csv(filename, index=False, encoding='utf_8_sig')
            print(f'数据已保存到 {filename}')
        except PermissionError:
            print(f'错误：没有权限写入文件 {filename}。请检查文件是否被其他程序占用或您是否有写入权限。')
        except Exception as e:
            print(f'保存文件时发生未知错误：{str(e)}')
    else:
        print('未获取到数据')
