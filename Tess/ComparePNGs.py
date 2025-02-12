import glob
import os
from PIL import Image

StarShadowAnalysis = glob.glob("./StarShadowAnalysis/*")
PNG = glob.glob("./PNG./*")

for index, folder in enumerate(StarShadowAnalysis):
    if os.path.exists(folder + "/" + os.path.basename(folder)[:-9] + "_eclipse_analysis_derivatives_h.png_ModelOnly.png"):
        starShadowPNG = folder + "/" + os.path.basename(folder)[:-9] + "_eclipse_analysis_derivatives_h.png_ModelOnly.png"

        CatalogID = os.path.basename(folder).split('_')[0]
        for indexPNG, justPNG in enumerate(PNG):
            if CatalogID in os.path.basename(justPNG):
                        image1 = Image.open(justPNG)
                        image2 = Image.open(starShadowPNG)

                        width = max(image1.width, image2.width)
                        height = image1.height + image2.height

                        combined_image = Image.new("RGBA", (width, height))

                        combined_image.paste(image1, (0, 0))
                        combined_image.paste(image2, (0, image1.height))

                        output_path = "./PNGCompare/" + CatalogID +".png"
                        combined_image.save(output_path)