from functools import reduce
import numpy as np
import pandas as pd


data_path = 'data/02_eng_clean.csv'
# selection_dict
selection_dict = {
    'Selection_Battery_Voltage': {3: 1, 2: 0, 1: 0, 0: 0},
    'Selection_Battery_Supply': {3: 1, 2: 0, 1: 0, 0: 0},
    'Selection_FCS': {3: 1, 2: 0, 1: 0, 0: 0},
    'Selection_I2O': {3: 1, 2: 0, 1: 1, 0: 0},
    'Selection_Power_Star': {3: 1, 2: 0, 1: 1, 0: 0},
    'Selection_Rectifier': {3: 1, 2: 1, 1: 0, 0: 0},
    'Selection_PV': {3: 1, 2: 1, 1: 0, 0: 0},
    'Selection_Battery_HighTemp': {3: 1, 2: 0, 1: 0, 0: 0}
}


def data_preprocessing(data_path, selection_dict=selection_dict):
    # %% function
    data_clean = pd.read_csv(data_path)

    # %% calculate kpis
    # multi-mapping
    def multi_mapping(df, col_list, dict_list):
        series = reduce(lambda x, y: x * y, [df[col].map(dict) for col, dict in zip(col_list, dict_list)])

        return series

    # multi-condition
    def multi_condition(series, select_dict):
        array = np.select(
            condlist=[series.to_frame(name='col').eval(''.join(['col', select_dict.get('cond')[i]])) for i in
                      range(len(select_dict.get('cond')))],
            choicelist=select_dict.get('choice'),
            default=select_dict.get('default'),
        )

        return array

    # Selection Condition
    def cal_Selection_Col(df, selection_dict, col_select, col_priority):
        df[col_select] = df[col_priority].map(selection_dict.get(col_select))

        return df

    # %% PV
    def cal_Upgrade_PV(df, col='Upgrade_PV', fillna=None):
        # avoid nan value
        df['Power_Supply_PV_Panels_Current'] = df['Power_Supply_PV_Panels_Current'].fillna(0)
        # df[col] = df.eval('(Power_Supply_PV_Out >= 1000 & Power_Supply_PV_Panels_Current == 0) * 1')
        df[col] = df.eval('(Power_Supply_PV_Out >= 969 & Power_Supply_PV_Panels_Current == 0) * 1')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Upgrade_PV(data_clean)

    def cal_Priority_PV(df, col='Priority_PV'):
        series = df['Power_Supply_PV_Out']
        # select_dict = {'cond': [' > 1825', ' > 1095', ' > 1000'], 'choice': [3, 2, 1], 'default': 0}
        select_dict = {'cond': [' > 998', ' > 987', ' > 969'], 'choice': [3, 2, 1], 'default': 0}

        df[col] = multi_condition(series, select_dict)

        return df

    data_clean = cal_Priority_PV(data_clean)

    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_PV', col_priority='Priority_PV')

    def cal_Power_Supply_PV_Annual_Tobe(df, col='Power_Supply_PV_Annual_Tobe'):
        df[col] = np.nanprod(
            [
                df['Power_Supply_PV_Out'],
                df['Power_Supply_PV_Panels_Tobe'],
                df['Power_Supply_PV_Specification_Tobe'],
                df['Selection_PV']
            ], axis=0
        ) * 0.825555 / 1e6

        return df

    data_clean = cal_Power_Supply_PV_Annual_Tobe(data_clean)

    # %% Rectifier
    def cal_Upgrade_Rectifier(df, col='Upgrade_Rectifier'):
        series = df['Rectifier_Efficiency_Current']
        select_dict = {'cond': [' > 0.95'], 'choice': [0], 'default': 1}

        df[col] = multi_condition(series, select_dict)

        return df

    data_clean = cal_Upgrade_Rectifier(data_clean)

    def cal_Priority_Rectifier(df, col='Priority_Rectifier'):
        series = df['Rectifier_Efficiency_Current']
        select_dict = {'cond': [' < 0.9', ' < 0.92', ' < 0.95'], 'choice': [3, 2, 1], 'default': 0}

        df[col] = multi_condition(series, select_dict)

        return df

    data_clean = cal_Priority_Rectifier(data_clean)

    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_Rectifier', col_priority='Priority_Rectifier')

    def cal_Rectifier_Efficiency_Tobe(df, col='Rectifier_Efficiency_Tobe'):
        series = df['Selection_Rectifier']
        select_dict = {'cond': [' == 1'], 'choice': [0.97], 'default': df['Rectifier_Efficiency_Current']}

        df[col] = multi_condition(series, select_dict)

        return df

    data_clean = cal_Rectifier_Efficiency_Tobe(data_clean)

    # %% Power Consumption
    def cal_Power_Consumption_Cable_Loss_Current(df, col='Power_Consumption_Cable_Loss_Current'):
        col_list = ['Rectifier_Model_Current', 'Battery_Type_Current']
        dict_list = [
            {'non-Huawei': 1, 'Huawei': 2},
            {'Lead-acid battery': 1, 'High-temperature lead-acid battery': 2,
             'Li-battery+lead-acid battery': 3,
             'Li-battery+high-temperature lead-acid battery': 4, 'Li-battery': 5}
        ]
        select_dict = {'cond': [' > 5'], 'choice': [0.068], 'default': 0.087}

        series = multi_mapping(df, col_list, dict_list)
        df[col] = multi_condition(series, select_dict)

        return df

    data_clean = cal_Power_Consumption_Cable_Loss_Current(data_clean)

    def cal_Power_Consumption_Cooling_Annual_Current(df, col='Power_Consumption_Cooling_Annual_Current',
                                                     fillna=None):
        df[col] = np.nansum([
            df['Cooling_Aircondition_Working_Hours_Current'] * df['Power_Consumption_Aircondition'],
            df['Cooling_FCS_Working_Hours_Current'] * df['Power_Consumption_FCS']
        ], axis=0) / 1e6

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Cooling_Annual_Current(data_clean)

    def cal_Power_Consumption_Cooling_Annual_Tobe(df, col='Power_Consumption_Cooling_Annual_Tobe', fillna=None):
        df[col] = np.nansum([
            df['Cooling_Aircondition_Working_Hours_Tobe'] * df['Power_Consumption_Aircondition'],
            df['Cooling_FCS_Working_Hours_Tobe'] * df['Power_Consumption_FCS']
        ], axis=0) / 1e6

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Cooling_Annual_Tobe(data_clean)

    def Power_Consumption_Cooling_Annual_Outdoor(df, col='Power_Consumption_Cooling_Annual_Outdoor', fillna=None):
        df[col] = np.nansum([
            df['Cooling_Aircondition_Working_Hours_Outdoor'] * df['Power_Consumption_Aircondition_Outdoor'],
            df['Cooling_FCS_Working_Hours_Outdoor'] * df['Power_Consumption_FCS_Outdoor']
        ], axis=0) / 1e6

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = Power_Consumption_Cooling_Annual_Outdoor(data_clean)

    # Battery Voltage
    def cal_Upgrade_Battery_Voltage(df, col='Upgrade_Battery_Voltage'):
        col_list = ['Battery_Type_Current']
        dict_list = [
            {'Lead-acid battery': 1,
             'High-temperature lead-acid battery': 1,
             'Li-battery+lead-acid battery': 0,
             'Li-battery+high-temperature lead-acid battery': 0,
             'Li-battery': 0}
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Upgrade_Battery_Voltage(data_clean)

    def cal_Priority_Battery_Voltage(df, col='Priority_Battery_Voltage'):
        col_list = ['Cooling_Type', 'Battery_Type_Current']
        dict_list = [
            {'Air condition': 3, 'Air condition+FCS': 3, 'FCS': 1},
            {'Lead-acid battery': 1,
             'High-temperature lead-acid battery': 1,
             'Li-battery+lead-acid battery': 0,
             'Li-battery+high-temperature lead-acid battery': 0,
             'Li-battery': 0}
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Priority_Battery_Voltage(data_clean)
    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_Battery_Voltage', col_priority='Priority_Battery_Voltage')

    # Battery Supply
    def cal_Upgrade_Battery_Supply(df, col='Upgrade_Battery_Supply'):
        col_list = ['Battery_Type_Current']
        dict_list = [
            {'Lead-acid battery': 1,
             'High-temperature lead-acid battery': 1,
             'Li-battery+lead-acid battery': 0,
             'Li-battery+high-temperature lead-acid battery': 0,
             'Li-battery': 0}
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Upgrade_Battery_Supply(data_clean)

    def cal_Priority_Battery_Supply(df, col='Priority_Battery_Supply'):
        col_list = ['Cooling_Type', 'Battery_Type_Current']
        dict_list = [
            {'Air condition': 3, 'Air condition+FCS': 3, 'FCS': 1},
            {'Lead-acid battery': 1,
             'High-temperature lead-acid battery': 1,
             'Li-battery+lead-acid battery': 0,
             'Li-battery+high-temperature lead-acid battery': 0,
             'Li-battery': 0}
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Priority_Battery_Supply(data_clean)
    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_Battery_Supply', col_priority='Priority_Battery_Supply')

    # Battery HighTemp
    def cal_Upgrade_Battery_HighTemp(df, col='Upgrade_Battery_HighTemp'):
        col_list = ['Battery_Type_Current']
        dict_list = [
            {'Lead-acid battery': 1,
             'High-temperature lead-acid battery': 0,
             'Li-battery+lead-acid battery': 1,
             'Li-battery+high-temperature lead-acid battery': 0,
             'Li-battery': 0}
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Upgrade_Battery_HighTemp(data_clean)

    def cal_Priority_Battery_HighTemp(df, col='Priority_Battery_HighTemp'):
        col_list = ['Priority_Battery_Supply', 'Cooling_Type', 'Battery_Type_Current']
        dict_list = [
            {3: 0, 2: 0, 1: 0, 0: 1},
            {'Air condition': 3, 'Air condition+FCS': 3, 'FCS': 1},
            {'Lead-acid battery': 1,
             'High-temperature lead-acid battery': 0,
             'Li-battery+lead-acid battery': 1,
             'Li-battery+high-temperature lead-acid battery': 0,
             'Li-battery': 0}
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Priority_Battery_HighTemp(data_clean)
    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_Battery_HighTemp',
                                   col_priority='Priority_Battery_HighTemp')

    # Power Consumption
    def cal_Power_Consumption_Cable_Loss_Tobe(df, col='Power_Consumption_Cable_Loss_Tobe', val=0.068):
        cond = np.nansum([df['Selection_Battery_Supply'], df['Selection_Battery_Voltage']], axis=0) > 0
        df[col] = np.select(condlist=[cond], choicelist=[val],
                            default=df['Power_Consumption_Cable_Loss_Current'])

        return df

    data_clean = cal_Power_Consumption_Cable_Loss_Tobe(data_clean)

    # I2O
    def cal_Upgrade_I2O(df, col='Upgrade_I2O'):
        col_list = ['Site_Room']
        dict_list = [
            {'Indoor': 1, 'Outdoor': 0},
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Upgrade_I2O(data_clean)

    def cal_Priority_I2O(df, col='Priority_I2O'):
        col_list = ['Site_Room', 'Cooling_Type']
        dict_list = [
            {'Indoor': 1, 'Outdoor': 0},
            {'Air condition': 3, 'Air condition+FCS': 3, 'FCS': 1},
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Priority_I2O(data_clean)

    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_I2O', col_priority='Priority_I2O')

    # FCS
    def cal_Upgrade_FCS(df, col='Upgrade_FCS'):
        col_list = ['Site_Room', 'Cooling_Type']
        dict_list = [
            {'Indoor': 1, 'Outdoor': 0},
            {'Air condition': 1, 'Air condition+FCS': 1, 'FCS': 0},
        ]

        df[col] = multi_mapping(df, col_list, dict_list)

        return df

    data_clean = cal_Upgrade_FCS(data_clean)

    def cal_Priority_FCS(df, col='Priority_FCS'):
        df[col] = df.eval('Priority_I2O < 3 & Upgrade_FCS == 1') * 3

        return df

    data_clean = cal_Priority_FCS(data_clean)
    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_FCS', col_priority='Priority_FCS')

    def cal_Power_Consumption_Cooling_Annual_Final(df, col='Power_Consumption_Cooling_Annual_Final'):
        cond = [df['Selection_I2O'] == 1, df['Selection_FCS'] == 1]
        val = [df['Power_Consumption_Cooling_Annual_Outdoor'], df['Power_Consumption_Cooling_Annual_Tobe']]
        df[col] = np.select(condlist=cond, choicelist=val,
                            default=df['Power_Consumption_Cooling_Annual_Current'])

        return df

    data_clean = cal_Power_Consumption_Cooling_Annual_Final(data_clean)

    # %% PowerStar
    def cal_Power_Consumption_Site_Radio_234G_Annual_Current(df,
                                                             col='Power_Consumption_Site_Radio_234G_Annual_Current',
                                                             fillna=None):
        df[col] = df.eval(
            'Power_Consumption_Site_Radio_234G * 24 * 365 * (1 - Power_Star_Activation * Power_Star_Efficiency_234G) / 1e6'
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Radio_234G_Annual_Current(data_clean)

    def cal_Priority_Power_Star(df, col='Priority_Power_Star'):
        series = df['Power_Star_Efficiency_234G']
        select_dict = {'cond': [' > 0.04', ' > 0.03'], 'choice': [3, 2], 'default': 1}

        df[col] = multi_condition(series, select_dict)

        return df

    data_clean = cal_Priority_Power_Star(data_clean)

    data_clean = cal_Selection_Col(data_clean, selection_dict,
                                   col_select='Selection_Power_Star', col_priority='Priority_Power_Star')

    def cal_Power_Star_Saving_234G_Annual(df, col='Power_Star_Saving_234G_Annual', fillna=None):
        df[col] = df.eval(
            'Power_Consumption_Site_Radio_234G_Annual_Current * Power_Star_Efficiency_234G * '
            '(1 + Power_Consumption_Cable_Loss_Tobe) / Rectifier_Efficiency_Tobe * Selection_Power_Star'
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Star_Saving_234G_Annual(data_clean)

    def cal_Power_Consumption_Site_Radio_5G_Annual(df, col='Power_Consumption_Site_Radio_5G_Annual', fillna=0):
        df[col] = df.eval(
            'Power_Consumption_Site_Radio_5G * 24 * 365 * (1 - Power_Star_Activation * Power_Star_Efficiency_234G) / 1e6'
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Radio_5G_Annual(data_clean)

    def cal_Power_Consumption_Site_Radio_2345G_Annual_Current(df,
                                                              col='Power_Consumption_Site_Radio_2345G_Annual_Current',
                                                              fillna=None):
        df[col] = np.nansum([
            df['Power_Consumption_Site_Radio_234G_Annual_Current'],
            df['Power_Consumption_Site_Radio_5G_Annual']
        ], axis=0)

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Radio_2345G_Annual_Current(data_clean)

    def cal_Power_Star_Saving_5G_Annual(df, col='Power_Star_Saving_5G_Annual', fillna=None):
        data_clean['Rectifier_Efficiency_Tobe'][0]

        df[col] = df.eval(
            '(Power_Consumption_Site_Radio_2345G_Annual_Current - Power_Consumption_Site_Radio_234G_Annual_Current) * '
            'Power_Star_Efficiency_5G * (1 + Power_Consumption_Cable_Loss_Tobe) / Rectifier_Efficiency_Tobe * '
            'Selection_Power_Star'
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Star_Saving_5G_Annual(data_clean)

    def cal_Power_Consumption_Site_Tower_5G_Annual(df, col='Power_Consumption_Site_Tower_5G_Annual', fillna=None):
        df[col] = np.nansum([
            df['Power_Consumption_Site_Tower_234G'], df['Power_Consumption_Site_Radio_5G']
        ], axis=0) * 24 * 365 / 1e6

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Tower_5G_Annual(data_clean)

    def cal_Power_Consumption_Site_Total_234G_Annual_Current(df,
                                                             col='Power_Consumption_Site_Total_234G_Annual_Current',
                                                             fillna=None):
        df[col] = df.eval(
            '(Power_Consumption_Site_Tower_234G * (1 + Power_Consumption_Cable_Loss_Current) + '
            '(Power_Consumption_Site_Radio_234G - Power_Consumption_Site_Tower_234G)) / '
            'Rectifier_Efficiency_Current * 24 * 365 / 1e6 + Power_Consumption_Cooling_Annual_Current'
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Total_234G_Annual_Current(data_clean)

    def cal_Power_Consumption_Site_Total_2345G_Annual_Current(df,
                                                              col='Power_Consumption_Site_Total_2345G_Annual_Current',
                                                              fillna=None):
        # to avoid nan issue
        df['temp'] = np.nansum(
            [df['Power_Consumption_Site_Tower_234G'], df['Power_Consumption_Site_Radio_5G']], axis=0
        )
        df[col] = df.eval(
            # '((Power_Consumption_Site_Tower_234G + Power_Consumption_Site_Radio_5G) * '
            '(temp * (1 + Power_Consumption_Cable_Loss_Current) + '
            '(Power_Consumption_Site_Radio_234G - Power_Consumption_Site_Tower_234G)) / '
            'Rectifier_Efficiency_Current * 24 * 365 / 1e6 + Power_Consumption_Cooling_Annual_Current'
        )
        df.drop('temp', axis=1, inplace=True)

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Total_2345G_Annual_Current(data_clean)

    def cal_Power_Consumption_Site_Radio_234G_Annual_Tobe(df, col='Power_Consumption_Site_Radio_234G_Annual_Tobe',
                                                          fillna=None):
        df[col] = np.nansum(
            [df['Power_Consumption_Site_Radio_234G_Annual_Current'], -df['Power_Star_Saving_234G_Annual']], axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Radio_234G_Annual_Tobe(data_clean)

    def cal_Power_Consumption_Site_Radio_2345G_Annual_Tobe(df, col='Power_Consumption_Site_Radio_2345G_Annual_Tobe',
                                                           fillna=None):
        df[col] = np.nansum(
            [
                df['Power_Consumption_Site_Radio_2345G_Annual_Current'],
                -df['Power_Star_Saving_234G_Annual'],
                -df['Power_Star_Saving_5G_Annual']
            ], axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Radio_2345G_Annual_Tobe(data_clean)

    def cal_Power_Consumption_Site_Total_234G_Annual_Tobe(df, col='Power_Consumption_Site_Total_234G_Annual_Tobe',
                                                          fillna=None):
        df[col] = df.eval(
            '(Power_Consumption_Site_Tower_234G * (1 + Power_Consumption_Cable_Loss_Tobe) + '
            '(Power_Consumption_Site_Radio_234G - Power_Consumption_Site_Tower_234G)) / '
            'Rectifier_Efficiency_Tobe * 24 * 365 / 1e6 + '
            'Power_Consumption_Cooling_Annual_Final - Power_Star_Saving_234G_Annual'
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_Power_Consumption_Site_Total_234G_Annual_Tobe(data_clean)

    def Power_Consumption_Site_Total_2345G_Annual_Tobe(df, col='Power_Consumption_Site_Total_2345G_Annual_Tobe',
                                                       fillna=None):
        # to avoid nan issue
        df['temp'] = np.nansum(
            [df['Power_Consumption_Site_Tower_234G'], df['Power_Consumption_Site_Radio_5G']], axis=0
        )
        df[col] = df.eval(
            # '((Power_Consumption_Site_Tower_234G + Power_Consumption_Site_Radio_5G) * '
            '(temp * (1 + Power_Consumption_Cable_Loss_Tobe) + '
            '(Power_Consumption_Site_Radio_234G - Power_Consumption_Site_Tower_234G)) / '
            'Rectifier_Efficiency_Tobe * 24 * 365 / 1e6 + '
            'Power_Consumption_Cooling_Annual_Final - Power_Star_Saving_234G_Annual - Power_Star_Saving_5G_Annual'
        )
        df.drop('temp', axis=1, inplace=True)

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = Power_Consumption_Site_Total_2345G_Annual_Tobe(data_clean)

    # KPI - 234G - Current
    def cal_KPI_TEE_234G_Current(df, col='KPI_TEE_234G_Current', fillna=None):
        df[col] = np.nanprod(
            [df['Data_Volume_234G_Month'] * 12, 1 / df['Power_Consumption_Site_Radio_234G_Annual_Current']],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_TEE_234G_Current(data_clean)

    def cal_KPI_SEE_234G_Current(df, col='KPI_SEE_234G_Current', fillna=None):
        df[col] = np.nanprod(
            [
                df['Power_Consumption_Site_Radio_234G_Annual_Current'],
                1 / df['Power_Consumption_Site_Total_234G_Annual_Current']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_SEE_234G_Current(data_clean)

    def cal_KPI_RER_234G_Current(df, col='KPI_RER_234G_Current', fillna=None):
        df[col] = np.nanprod(
            [
                df['Power_Supply_PV_Annual_Current'],
                1 / df['Power_Consumption_Site_Total_234G_Annual_Current']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_RER_234G_Current(data_clean)

    def cal_KPI_NCI_234G_Current(df, col='KPI_NCI_234G_Current', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_234G_Current) / (KPI_TEE_234G_Current * KPI_SEE_234G_Current)')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NCI_234G_Current(data_clean)

    def cal_KPI_NC_234G_Current(df, col='KPI_NC_234G_Current', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_234G_Current) * Power_Consumption_Site_Total_234G_Annual_Current')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NC_234G_Current(data_clean)

    # KPI - 2345G - Current
    def cal_KPI_TEE_2345G_Current(df, col='KPI_TEE_2345G_Current', fillna=None):
        df[col] = np.nanprod(
            [
                np.nansum([df['Data_Volume_234G_Month'], df['Data_Volume_5G_Month']], axis=0) * 12,
                1 / df['Power_Consumption_Site_Radio_2345G_Annual_Current']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_TEE_2345G_Current(data_clean)

    def cal_KPI_SEE_2345G_Current(df, col='KPI_SEE_2345G_Current', fillna=None):
        df[col] = np.nanprod(
            [
                df['Power_Consumption_Site_Radio_2345G_Annual_Current'],
                1 / df['Power_Consumption_Site_Total_2345G_Annual_Current']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_SEE_2345G_Current(data_clean)

    def cal_KPI_RER_2345G_Current(df, col='KPI_RER_2345G_Current', fillna=None):
        df[col] = np.nanprod(
            [
                df['Power_Supply_PV_Annual_Current'],
                1 / df['Power_Consumption_Site_Total_2345G_Annual_Current']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_RER_2345G_Current(data_clean)

    def cal_KPI_NCI_2345G_Current(df, col='KPI_NCI_2345G_Current', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_2345G_Current) / (KPI_TEE_2345G_Current * KPI_SEE_2345G_Current)')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NCI_2345G_Current(data_clean)

    def cal_KPI_NC_2345G_Current(df, col='KPI_NC_2345G_Current', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_2345G_Current) * Power_Consumption_Site_Total_2345G_Annual_Current')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NC_2345G_Current(data_clean)

    # KPI - 234G - Tobe
    def cal_KPI_TEE_234G_Tobe(df, col='KPI_TEE_234G_Tobe', fillna=None):
        df[col] = np.nanprod(
            [df['Data_Volume_234G_Month'] * 12, 1 / df['Power_Consumption_Site_Radio_234G_Annual_Tobe']],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_TEE_234G_Tobe(data_clean)

    def cal_KPI_SEE_234G_Tobe(df, col='KPI_SEE_234G_Tobe', fillna=None):
        df[col] = np.nanprod(
            [
                df['Power_Consumption_Site_Radio_234G_Annual_Tobe'],
                1 / df['Power_Consumption_Site_Total_234G_Annual_Tobe']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_SEE_234G_Tobe(data_clean)

    def cal_KPI_RER_234G_Tobe(df, col='KPI_RER_234G_Tobe', fillna=None):
        cond = [df['Selection_PV'] == 0]
        val = [df['Power_Supply_PV_Annual_Current']]
        dft = df['Power_Supply_PV_Annual_Tobe']

        df[col] = np.nanprod(
            [
                np.select(condlist=cond, choicelist=val, default=dft),
                1 / df['Power_Consumption_Site_Total_234G_Annual_Tobe']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_RER_234G_Tobe(data_clean)

    def cal_KPI_NCI_234G_Tobe(df, col='KPI_NCI_234G_Tobe', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_234G_Tobe) / (KPI_TEE_234G_Tobe * KPI_SEE_234G_Tobe)')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NCI_234G_Tobe(data_clean)

    def cal_KPI_NC_234G_Tobe(df, col='KPI_NC_234G_Tobe', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_234G_Tobe) * Power_Consumption_Site_Total_234G_Annual_Tobe')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NC_234G_Tobe(data_clean)

    # KPI - 2345G - Tobe
    def cal_KPI_TEE_2345G_Tobe(df, col='KPI_TEE_2345G_Tobe', fillna=None):
        df[col] = np.nanprod(
            [
                np.nansum([df['Data_Volume_234G_Month'], df['Data_Volume_5G_Month']], axis=0) * 12,
                1 / df['Power_Consumption_Site_Radio_2345G_Annual_Tobe']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_TEE_2345G_Tobe(data_clean)

    def cal_KPI_SEE_2345G_Tobe(df, col='KPI_SEE_2345G_Tobe', fillna=None):
        df[col] = np.nanprod(
            [
                df['Power_Consumption_Site_Radio_2345G_Annual_Tobe'],
                1 / df['Power_Consumption_Site_Total_2345G_Annual_Tobe']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_SEE_2345G_Tobe(data_clean)

    def cal_KPI_RER_2345G_Tobe(df, col='KPI_RER_2345G_Tobe', fillna=None):
        cond = [df['Selection_PV'] == 0]
        val = [df['Power_Supply_PV_Annual_Current']]
        dft = df['Power_Supply_PV_Annual_Tobe']

        df[col] = np.nanprod(
            [
                np.select(condlist=cond, choicelist=val, default=dft),
                1 / df['Power_Consumption_Site_Total_2345G_Annual_Tobe']
            ],
            axis=0
        )

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_RER_2345G_Tobe(data_clean)

    def cal_KPI_NCI_2345G_Tobe(df, col='KPI_NCI_2345G_Tobe', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_2345G_Tobe) / (KPI_TEE_2345G_Tobe * KPI_SEE_2345G_Tobe)')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NCI_2345G_Tobe(data_clean)

    def cal_KPI_NC_2345G_Tobe(df, col='KPI_NC_2345G_Tobe', fillna=None):
        df[col] = df.eval('Alpha * (1 - KPI_RER_2345G_Tobe) * Power_Consumption_Site_Total_2345G_Annual_Tobe')

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        return df

    data_clean = cal_KPI_NC_2345G_Tobe(data_clean)

    # Power Saving
    def cal_Power_Saving_PV(df, col='Power_Saving_PV'):
        df[col] = df['Power_Supply_PV_Annual_Tobe']
        return df

    def cal_Power_Saving_Rectifier(df, col='Power_Saving_Rectifier'):
        df[col] = df.eval(
            'Power_Consumption_Site_Radio_2345G_Annual_Current * '
            '(1 / Rectifier_Efficiency_Current - 1 / Rectifier_Efficiency_Tobe)'
        )
        return df

    def cal_Power_Saving_Cable(df, col='Power_Saving_Cable'):
        df[col] = df.eval(
            'Power_Consumption_Site_Radio_2345G_Annual_Current * '
            '(Power_Consumption_Cable_Loss_Current - Power_Consumption_Cable_Loss_Tobe)'
        )
        return df

    def cal_Power_Saving_FCS(df, col='Power_Saving_FCS'):
        df[col] = df.eval(
            'Selection_FCS * (Power_Consumption_Cooling_Annual_Current - Power_Consumption_Cooling_Annual_Tobe)'
        )
        return df

    def cal_Power_Saving_I2O(df, col='Power_Saving_I2O'):
        df[col] = df.eval(
            'Selection_I2O * (Power_Consumption_Cooling_Annual_Current - Power_Consumption_Cooling_Annual_Outdoor)'
        )
        return df

    def cal_Power_Saving_PowerStar(df, col='Power_Saving_PowerStar'):
        df[col] = df.eval(
            'Power_Star_Saving_234G_Annual + Power_Star_Saving_5G_Annual'
        )
        return df

    def cal_Power_Saving_Total(df, col='Power_Saving_Total'):
        df[col] = np.nansum([
            df['Power_Saving_PV'], df['Power_Saving_Rectifier'], df['Power_Saving_Cable'],
            df['Power_Saving_FCS'], df['Power_Saving_I2O'], df['Power_Saving_PowerStar']
        ], axis=0)

        return df

    data_clean = cal_Power_Saving_PV(data_clean)
    data_clean = cal_Power_Saving_Rectifier(data_clean)
    data_clean = cal_Power_Saving_Cable(data_clean)
    data_clean = cal_Power_Saving_FCS(data_clean)
    data_clean = cal_Power_Saving_I2O(data_clean)
    data_clean = cal_Power_Saving_PowerStar(data_clean)
    data_clean = cal_Power_Saving_Total(data_clean)

    def cal_Electricity_Saving_Total(df, electricity_price_per_kwh=0.168, col='Electricity_Saving_Total'):
        df[col] = df['Power_Saving_Total'] * 1000 * electricity_price_per_kwh
        return df

    def cal_Rental_Saving(df, rental_price_per_year=500, col='Rental_Saving'):
        df[col] = df['Selection_I2O'] * rental_price_per_year
        return df

    def cal_Opex_Saving_Total(df, opex_cost_per_year=1560, col='Opex_Saving_Total'):
        df[col] = df['Selection_Battery_Supply'] * opex_cost_per_year * 0.02 + \
                  df['Selection_I2O'] * opex_cost_per_year * 0.05
        return df

    def cal_Carbon_Saving(df, carbon_price_per_tco2=82, col='Carbon_Saving'):
        df[col] = df.eval(
            '(KPI_NC_2345G_Current - KPI_NC_2345G_Tobe) * Alpha / 1e6 * @carbon_price_per_tco2'
        )
        return df

    data_clean = cal_Electricity_Saving_Total(data_clean)
    data_clean = cal_Rental_Saving(data_clean)
    data_clean = cal_Opex_Saving_Total(data_clean)
    data_clean = cal_Carbon_Saving(data_clean)

    # Summary
    # non-zero statistics
    # print(data_clean.filter(regex='^Power_Saving').replace(0, np.nan).describe())

    return data_clean


# df = data_preprocessing(data_path, selection_dict=selection_dict)
# df[['Electricity_Saving_Total', 'Rental_Saving', 'Opex_Saving_Total', 'Carbon_Saving']].sum()
