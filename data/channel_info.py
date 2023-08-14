def get_channel_info(tissue):
    if tissue == 'CRC':
        channels = [
            #Round 0
            "DAPI",
            "Control",
            "Control",
            "Control",
            #Round 1
            "DAPI_2",
            "CD3",
            "NaKATPase",
            "CD45RO",
            #Round 2
            "DAPI_3",
            "Ki67",
            "PanCK",
            "aSMA",
            #Round 3
            "DAPI_4",
            "CD4",
            "CD45",
            "PD-1/CD279",
            #Round 4
            "DAPI_5",
            "CD20",
            "CD68",
            "CD8a",
            #Round 5
            "DAPI_6",
            "CD163",
            "FOXP3",
            "PD-L1/CD274",
            #Round 6
            "DAPI_7",
            "E-Cadherin",
            "Vimentin",
            "CDX2",
            #Round 7
            "DAPI_8",
            "LaminABC",
            "Desmin",
            "CD31",
            #Round 8
            "DAPI_9",
            "PCNA",
            "Ki67",
            "Collagen IV",
        ]

        keep_channels = ["DAPI"] + [ch for ch in channels if ch != "Control" and not ch.startswith('DAPI')] + ["DAPI_9"]
        keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
        ch2idx = {ch:i for i,ch in enumerate(keep_channels)}
        return keep_channels, keep_channels_idx, ch2idx
        