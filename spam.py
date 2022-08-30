import mailbox
import emoji
import re
from bs4 import BeautifulSoup

msgtextfinal = []


def give_emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text)

msg = mailbox.mbox('datas/Spam.mbox')

def message(text):
    try:
        soup = BeautifulSoup(text)
        msgtxt = soup.text.split('\n')
        for isp in msgtxt:
            if isp != '':
                isp = give_emoji_free_text(isp)
                isp = re.sub('[^a-zA-Zа-яА-Я0-9]+[é]+[à]+[&]+[ç]', ' ', isp)
                msgtextfinal.append((isp + "\n"))
        print(msgtextfinal)
        return msgtextfinal
    except:
        print("ko")

cont = 25
for i in msg.values():

    mail_content = ''
    mail_content = i.get_payload()
    msgtextfinal = []
    if isinstance(mail_content, list) == False:
        msgtextfinal.append("from : " + str(i['from']) + "\n")
        msgtextfinal.append("subject : "+str(i['subject']))
        f = open("spam/Sp2am" + str(cont) + ".txt", "w")
        f.writelines(message(mail_content))
        f.close()
        cont += 1