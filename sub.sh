export KAGGLE_USERNAME=ankitsajwan
export KAGGLE_KEY=4089d05f2b8cfb00d832021fa3fdf94c
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions submissions seti-breakthrough-listen
kaggle competitions submit seti-breakthrough-listen -f SETI/output/SeResNet_legacy_seresnet18_bs32_AllChl256258_mixupTrue_augSwapDropFlip_ohemFalse_scdCosineAnnealingWarmRestarts_dropoutFalse_InvOrigNorm_epoch50/cv0.9899636725625476_auc_SeResNet_legacy_seresnet18_bs32_AllChl256258_mixupTrue_augSwapDropFlip_ohemFalse_scdCosineAnnealingWarmRestarts_dropoutFalse_InvOrigNorm_epoch50 -m 'cv0.9899636725625476_auc_SeResNet_legacy_seresnet18_bs32_AllChl256258_mixupTrue_augSwapDropFlip_ohemFalse_scdCosineAnnealingWarmRestarts_dropoutFalse_InvOrigNorm_epoch50'