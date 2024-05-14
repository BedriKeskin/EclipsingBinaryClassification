# Gaia'nın gaiadr3.vari_eclipsing_binary tablosundaki bütün 2.1 milyon yıldızın
# listesini indirir (ama sorgu sonucu 2000 adet dönüyor, sınırlandırmışlar)
# ve ışık eğrilerini tek tek indirip LCdata olarak kaydeder.

import datetime
import os

import requests
from astroquery.gaia import Gaia

now1 = datetime.datetime.now()
print("Start time: ", now1)

if not os.path.exists("LCdata"):
    os.makedirs("LCdata")

job = Gaia.launch_job("SELECT * FROM gaiadr3.vari_eclipsing_binary")
results = job.get_results()
print(len(results))

for index, result in enumerate(results):
    print(index, result['source_id'])
    url = 'https://gea.esac.esa.int/data-server/data?ID=Gaia+DR3+' + str(
        result['source_id']) + '&RETRIEVAL_TYPE=EPOCH_PHOTOMETRY&VALID_DATA=true'
    print(url)
    byte = requests.get(url).content

    with open('LCdata/' + str(result['source_id']) + '.xml', 'wb') as file:
        file.write(byte)
        file.close()

now2 = datetime.datetime.now()
print("End time: ", now2)
print("Elapsed time: ", now2 - now1)
