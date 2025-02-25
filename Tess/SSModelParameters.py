import star_shadow as sts
import glob
import os
import pandas as pd
import csv


for root, _, files in os.walk("StarShadowAnalysis"):
    for file in files:
        if file.endswith("_summary.csv"):
            print(os.path.join(root, file))
            with open(os.path.join(root, file), newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip() == "stage":
                        stage = row[1].strip()
                    if row and row[0].strip() == "t_mean":
                        t_mean = row[1].strip()
                    if row and row[0].strip() == "period":
                        period = row[1].strip()
                    if row and row[0].strip() == "t_1":
                        t_1 = row[1].strip()
                    if row and row[0].strip() == "t_2":
                        t_2 = row[1].strip()
                    if row and row[0].strip() == "t_1_1":
                        t_1_1 = row[1].strip()
                    if row and row[0].strip() == "t_1_2":
                        t_1_2 = row[1].strip()
                    if row and row[0].strip() == "t_2_1":
                        t_2_1 = row[1].strip()
                    if row and row[0].strip() == "t_2_2":
                        t_2_2 = row[1].strip()
                    if row and row[0].strip() == "t_b_1_1":
                        t_b_1_1 = row[1].strip()
                    if row and row[0].strip() == "t_b_1_2":
                        t_b_1_2 = row[1].strip()
                    if row and row[0].strip() == "t_b_2_1":
                        t_b_2_1 = row[1].strip()
                    if row and row[0].strip() == "t_b_2_2":
                        t_b_2_2 = row[1].strip()
                    if row and row[0].strip() == "depth_1":
                        depth_1 = row[1].strip()
                    if row and row[0].strip() == "depth_2":
                        depth_2 = row[1].strip()
                    if row and row[0].strip() == "ecosw_form":
                        ecosw_form = row[1].strip()
                    if row and row[0].strip() == "esinw_form":
                        esinw_form = row[1].strip()
                    if row and row[0].strip() == "cosi_form":
                        cosi_form = row[1].strip()
                    if row and row[0].strip() == "phi_0_form":
                        phi_0_form = row[1].strip()
                    if row and row[0].strip() == "log_rr_form":
                        log_rr_form = row[1].strip()
                    if row and row[0].strip() == "log_sb_form":
                        log_sb_form = row[1].strip()
                    if row and row[0].strip() == "e_form":
                        e_form = row[1].strip()
                    if row and row[0].strip() == "w_form":
                        w_form = row[1].strip()
                    if row and row[0].strip() == "i_form":
                        i_form = row[1].strip()
                    if row and row[0].strip() == "r_sum_form":
                        r_sum_form = row[1].strip()
                    if row and row[0].strip() == "r_rat_form":
                        r_rat_form = row[1].strip()
                    if row and row[0].strip() == "sb_rat_form":
                        sb_rat_form = row[1].strip()
                    if row and row[0].strip() == "ecosw_phys":
                        ecosw_phys = row[1].strip()
                    if row and row[0].strip() == "esinw_phys":
                        esinw_phys = row[1].strip()
                    if row and row[0].strip() == "cosi_phys":
                        cosi_phys = row[1].strip()
                    if row and row[0].strip() == "phi_0_phys":
                        phi_0_phys = row[1].strip()
                    if row and row[0].strip() == "log_rr_phys":
                        log_rr_phys = row[1].strip()
                    if row and row[0].strip() == "log_sb_phys":
                        log_sb_phys = row[1].strip()
                    if row and row[0].strip() == "e_phys":
                        e_phys = row[1].strip()
                    if row and row[0].strip() == "w_phys":
                        w_phys = row[1].strip()
                    if row and row[0].strip() == "i_phys":
                        i_phys = row[1].strip()
                    if row and row[0].strip() == "r_sum_phys":
                        r_sum_phys = row[1].strip()
                    if row and row[0].strip() == "r_rat_phys":
                        r_rat_phys = row[1].strip()
                    if row and row[0].strip() == "sb_rat_phys":
                        sb_rat_phys = row[1].strip()
                    if row and row[0].strip() == "ratio_1_1":
                        ratio_1_1 = row[1].strip()
                    if row and row[0].strip() == "ratio_1_2":
                        ratio_1_2 = row[1].strip()
                    if row and row[0].strip() == "ratio_2_1":
                        ratio_2_1 = row[1].strip()
                    if row and row[0].strip() == "ratio_2_2":
                        ratio_2_2 = row[1].strip()
                    if row and row[0].strip() == "ratio_3_1":
                        ratio_3_1 = row[1].strip()
                    if row and row[0].strip() == "ratio_3_2":
                        ratio_3_2 = row[1].strip()
                    if row and row[0].strip() == "ratio_4_1":
                        ratio_4_1 = row[1].strip()
                    if row and row[0].strip() == "ratio_4_2":
                        ratio_4_2 = row[1].strip()

                print("ratio_4_2 ", ratio_4_2, " stage " , stage)

