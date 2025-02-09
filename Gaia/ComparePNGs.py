import glob
import os
from PIL import Image

StarShadowAnalysis = glob.glob("./StarShadowAnalysis/*")
PNG = glob.glob("./PNG./*")
PNGMowlaviFit = glob.glob("./PNGMowlaviFit/*")

for index, folder in enumerate(StarShadowAnalysis):
    if os.path.exists(folder + "/" + os.path.basename(folder)[:-9] + "_eclipse_analysis_derivatives_h.png_ModelOnly.png"):
        starShadowPNG = folder + "/" + os.path.basename(folder)[:-9] + "_eclipse_analysis_derivatives_h.png_ModelOnly.png"

        GaiaID = os.path.basename(folder).split('_')[0]
        for indexPNG, justPNG in enumerate(PNG):
            if GaiaID in os.path.basename(justPNG):
                for indexMowlaviPNG, mowlaviPNG in enumerate(PNGMowlaviFit):
                    if GaiaID in os.path.basename(mowlaviPNG):
                        image1 = Image.open(justPNG)
                        image2 = Image.open(mowlaviPNG)
                        image3 = Image.open(starShadowPNG)

                        # Resimlerin boyutlarını kontrol et ve en yükseği al (üst üste yerleştirilecek)
                        width = max(image1.width, image2.width, image3.width)
                        height = image1.height + image2.height + image3.height

                        # Yeni bir boş resim oluştur (en geniş ve toplam yükseklik kadar)
                        combined_image = Image.new("RGBA", (width, height))

                        # Resimleri üst üste yapıştır
                        combined_image.paste(image1, (0, 0))
                        combined_image.paste(image2, (0, image1.height))
                        combined_image.paste(image3, (0, image1.height + image2.height))

                        # Sonucu kaydet
                        output_path = "./PNGCompare/" + GaiaID +".png"
                        combined_image.save(output_path)