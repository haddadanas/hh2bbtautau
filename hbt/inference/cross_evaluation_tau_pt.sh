# for j in 35 36 37 38 40 45 50 80; do law run cf.ProduceColumnsWrapper --configs 22pre_v14 --version sel_cut_limit_v2 --producers ml_selection_${j} --workers 8 --datasets my; done
# for i in 35 36 37 38 40 45 50 80; do for j in 35 36 37 38 40 45 50 80; do law run cf.CreateDatacards --config 22pre_v14 --version sel_cut_limit_v2 --producers ml_selection_${i} --inference-model tau_pt${j} --workers 14; done; done


result=""
label=""
# i=35
for i in 35 36 37 38 40 45 50 80; do
  # for j in 35 36 37 38 40 45 50 80; do
  j=$i
        result+=/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__tautau/red__tautau/prod__ml_selection_${i}/hist__default/inf__tau_pt${j}/sel_cut_limit_v2/datacard__tautau__ml_selected_50__tau_pt_${j}.txt
        # result+=/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__tautau/red__tautau/prod__ml_selection_${i}/hist__default/inf__tau_pt${j}/sel_cut_limit_v2/datacard__tautau__signal__tau_pt_${j}.txt
        # result+=/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__tautau/red__tautau/prod__ml_selection_${i}/hist__default/inf__tau_pt${j}/sel_cut_limit_v3/datacard__tautau__signal__tau_pt_${j}__hooks_flats_kl1_n10_guarded.txt
        # result+=/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__tautau/red__tautau/prod__ml_selection_${i}/hist__default/inf__tau_pt${j}/sel_cut_limit_v3/datacard__tautau__ml_selected_50__tau_pt_${j}__hooks_flats_kl1_n10_guarded.txt
        # result+=/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__default/prod__ml_selection_${i}/weight__default/inf__tau_pt${j}/sel_cut_limit_v3/datacard__tautau__ml_selected_50__tau_pt_${j}__bin_dnn_signal.txt
        # result+=/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__default/prod__ml_selection_${i}/weight__default/inf__tau_pt${j}/v2/datacard__tautau__ml_selected_50__tau_pt_${j}__hh_mass.txt
        # result+=/data/dust/user/haddadan/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__default/prod__ml_selection_${i}/weight__default/inf__tau_pt${j}/sel_cut_limit_v2/datacard__tautau__signal__1bjet__tau_pt_${j}__bin_dnn_signal.txt
        # result+=/data/dust/user/haddadan/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__default/prod__ml_selection_${i}/weight__default/inf__sel_tau_pt${j}/sel_cut_limit_v2/datacard__tautau__signal__1bjet__tau_pt_${j}__hh_mass.txt
        label+="E ${j} T ${i}"
        if [ $j -ne 80 ] || [ $i -ne 80 ]; then
            result+=":"
            label+=","
        fi
  # done
done

echo "$label"
echo "$result"
law run PlotUpperLimitsAtPoint --datacard-names "$label"  --pois r  --file-types pdf --version v2 --view-cmd imgcat --multi-datacards "$result" --workers 8 --save-plot-data


# result=""
# label=""
# i=35
#   for j in 35 36 37 38 40 45 50 80; do
#         result+=/data/dust/user/haddadan/hbt_store/analysis_hbt/cf.CreateDatacards/22pre_v14/calib__default/sel__default/prod__ml_selection_${i}/weight__default/inf__sel_tau_pt${j}/sel_cut_limit_v2/datacard__tautau__signal__1bjet__tau_pt_${j}__hh_mass.txt
#         label+="sel${j}_mod${i}"
#         if [ $j -ne 80 ]; then
#             result+=":"
#             label+=","
#         fi
#   done

# echo "$label"
# echo "$result"
# law run PlotUpperLimitsAtPoint --datacard-names "$label" --show-theory False --pois r  --file-types pdf --version sel_cut_limit_v2 --view-cmd imgcat --multi-datacards "$result" --workers 8 --save-plot-data --save-hep-data --x-min 100 --plot-postfix test