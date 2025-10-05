import requests
from bs4.element import PageElement

cookies = {
    'session-id': '140-7307879-8660869',
    'session-id-time': '2082787201l',
    'ad-oo': '0',
    'ci': 'eyJpc0dkcHIiOmZhbHNlfQ',
    'ubid-main': '134-1928029-4806339',
    'session-token': 'hDHHQM31xya4A5VshRfwLV6Sp8akEG/Bq9+oz/AiFK8AfwOXApo4dEVS9qeCcFQKIiCYx3nyk4kgXtw8azbcwt63VMvTT+pt9lXBNBwGiPH4PZ7/p+IzeXB7MqmY6oTpnatzBzj+BPqpHUCv6PqihUb2uCVjuXNMkBtPyoFbJwqQg0c32n9Cy8VrHmiPQnfTHyHlVks+IDqRskZ42dbK4HdPHzN6DY84RzbubDScOn8P9h4uArxAYTuCf3rtt3dIzTEYLMZtR7Uf8tWNCo75YoQOsiUUphM10uoyNQvoCU68ARgqqBLqVhWGkO7nRf2LpXRDhaCzsEWFNjse48xpbcT9j4j/EieM',
    'csm-hit': 'adb:adblk_no&t:1759451453309&tb:GDHJA7T02TFCJ4QS51HG+s-ZG1Z0HD6YBKRCZFP0XEM|1759451453309',
    'aws-waf-token': '24295340-6e3b-4765-b9fc-81d12e86369f:BgoArh4Cq2Y3AAAA:DYaVam7voUfTxnOhkMmSeVjwjy6YqC8GqURWh7vasIs8LOxdZxoa16wQhsFv7aqVghzoskFLFU/E7tqdnCa8qnxydVBFDVKyBGGcvH+sn1EyhJTDiQH6aKPfNYykDqWw736f+cIvq9NOP3NElEf4YVZe2GqilSwYJjaCvI8iXTMeCTnyUnkGlYIiSwKBOPQ=',
}

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9,ja;q=0.8,vi;q=0.7',
    'cache-control': 'no-cache',
    'priority': 'u=0, i',
    'referer': 'https://www.imdb.com/chart/top/?ref_=nv_mv_250_6',
    'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
    # 'cookie': 'session-id=140-7307879-8660869; session-id-time=2082787201l; ad-oo=0; ci=eyJpc0dkcHIiOmZhbHNlfQ; ubid-main=134-1928029-4806339; session-token=hDHHQM31xya4A5VshRfwLV6Sp8akEG/Bq9+oz/AiFK8AfwOXApo4dEVS9qeCcFQKIiCYx3nyk4kgXtw8azbcwt63VMvTT+pt9lXBNBwGiPH4PZ7/p+IzeXB7MqmY6oTpnatzBzj+BPqpHUCv6PqihUb2uCVjuXNMkBtPyoFbJwqQg0c32n9Cy8VrHmiPQnfTHyHlVks+IDqRskZ42dbK4HdPHzN6DY84RzbubDScOn8P9h4uArxAYTuCf3rtt3dIzTEYLMZtR7Uf8tWNCo75YoQOsiUUphM10uoyNQvoCU68ARgqqBLqVhWGkO7nRf2LpXRDhaCzsEWFNjse48xpbcT9j4j/EieM; csm-hit=adb:adblk_no&t:1759451453309&tb:GDHJA7T02TFCJ4QS51HG+s-ZG1Z0HD6YBKRCZFP0XEM|1759451453309; aws-waf-token=24295340-6e3b-4765-b9fc-81d12e86369f:BgoArh4Cq2Y3AAAA:DYaVam7voUfTxnOhkMmSeVjwjy6YqC8GqURWh7vasIs8LOxdZxoa16wQhsFv7aqVghzoskFLFU/E7tqdnCa8qnxydVBFDVKyBGGcvH+sn1EyhJTDiQH6aKPfNYykDqWw736f+cIvq9NOP3NElEf4YVZe2GqilSwYJjaCvI8iXTMeCTnyUnkGlYIiSwKBOPQ=',
}

params = {
    'ref_': 'nv_mv_250_6',
}

response = requests.get('https://www.imdb.com/chart/top/', params=params, cookies=cookies, headers=headers)

from bs4 import BeautifulSoup

soup = BeautifulSoup(response.content, "html.parser")

all_li_tags = soup.find_all("li", class_="ipc-metadata-list-summary-item")

results = []

for i, li_tag in enumerate(all_li_tags):
    div_tag = li_tag.find("div", class_="ipc-media ipc-media--poster-27x40 ipc-image-media-ratio--poster-27x40 ipc-media--media-radius ipc-media--base ipc-media--poster-s ipc-poster__poster-image ipc-media__img") # type: ignore
    img_tag = div_tag.find("img") # type: ignore
    img_url = img_tag['src'] # type: ignore

    h3_tag = li_tag.find("h3", class_="ipc-title__text ipc-title__text--reduced") # type: ignore
    title = h3_tag.text # type: ignore

    span_tag = li_tag.find_all("span", class_="sc-15ac7568-7 cCsint cli-title-metadata-item") # type: ignore
    year = span_tag[0].text
    hour = span_tag[1].text

    span_tag = li_tag.find("span", class_="ipc-rating-star--rating") # type: ignore
    rate = span_tag.text # type: ignore
    
    res_object = {
        "id": i,
        "image_url": img_url,
        "title": title,
        "year": year,
        "hour": hour,
        "rate": rate
    }

    print(res_object)
    results.append(res_object)


print(results)
