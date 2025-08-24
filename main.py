#importing url opener
from urllib.request import urlopen, Request
#importing web scrapers
from bs4 import BeautifulSoup
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#allows us to manipulate data in table structure
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Streamlit page config must be set before any other Streamlit commands
try:
    st.set_page_config(
        page_title="Stockfeels.com",
        page_icon="https://i.postimg.cc/PrPKRBTq/Foto-COMERCIANDOLA.png",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except Exception:
    # If called multiple times in interactive reloads, ignore
    pass
from datetime import date as date2
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
import re
import json
import os
from pathlib import Path
from openai import OpenAI
from firecrawl import FirecrawlApp
import logging
import traceback

# Simple file + in-memory logging for quick debugging
LOG_FILE = '.stockfeels_debug.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

def log_debug(msg: str, exc_info: bool = False):
    """Log to file and keep a short in-memory buffer in Streamlit session for display."""
    try:
        if exc_info:
            logging.exception(msg)
        else:
            logging.debug(msg)
    except Exception:
        pass
    try:
        # store last messages in session for quick UI inspection
        if 'debug_logs' not in st.session_state:
            st.session_state['debug_logs'] = []
        entry = msg if not exc_info else f"{msg}\n" + traceback.format_exc()
        st.session_state['debug_logs'].append(entry)
        # keep only last 200 entries
        st.session_state['debug_logs'] = st.session_state['debug_logs'][-200:]
    except Exception:
        # fail silently if Streamlit session not available
        pass

# Optional: embed filtros.json content here for deployments where reading an external
# file is inconvenient (for example, some hosting platforms). To use, paste the
# full JSON content of `filtros.json` between the triple quotes below and save
# the file. Leave the string empty ("""") to keep loading from disk.
EMBEDDED_FINVIZ_FILTERS_JSON = """
{
    "exchanges": {
      "NYSE": "https://finviz.com/screener.ashx?v=111&f=exch_nyse",
      "AMEX": "https://finviz.com/screener.ashx?v=111&f=exch_amex",
      "Nasdaq": "https://finviz.com/screener.ashx?v=111&f=exch_nasd"
    },
    "market_caps": {
      "Mega": "https://finviz.com/screener.ashx?v=111&f=cap_mega",
      "Large": "https://finviz.com/screener.ashx?v=111&f=cap_large",
      "Mid": "https://finviz.com/screener.ashx?v=111&f=cap_mid",
      "Small": "https://finviz.com/screener.ashx?v=111&f=cap_small",
      "Micro": "https://finviz.com/screener.ashx?v=111&f=cap_micro",
      "Nano": "https://finviz.com/screener.ashx?v=111&f=cap_nano",
      "LargeOver": "https://finviz.com/screener.ashx?v=111&f=cap_largeover",
      "MidOver": "https://finviz.com/screener.ashx?v=111&f=cap_midover",
      "SmallOver": "https://finviz.com/screener.ashx?v=111&f=cap_smallover",
      "MicroOver": "https://finviz.com/screener.ashx?v=111&f=cap_microover",
      "LargeUnder": "https://finviz.com/screener.ashx?v=111&f=cap_largeunder",
      "MidUnder": "https://finviz.com/screener.ashx?v=111&f=cap_midunder",
      "SmallUnder": "https://finviz.com/screener.ashx?v=111&f=cap_smallunder",
      "MicroUnder": "https://finviz.com/screener.ashx?v=111&f=cap_microunder"
    },
    "earnings_dates": {
      "Today": "https://finviz.com/screener.ashx?v=111&f=earningsdate_today",
      "TodayBefore": "https://finviz.com/screener.ashx?v=111&f=earningsdate_todaybefore",
      "TodayAfter": "https://finviz.com/screener.ashx?v=111&f=earningsdate_todayafter",
      "Tomorrow": "https://finviz.com/screener.ashx?v=111&f=earningsdate_tomorrow",
      "TomorrowBefore": "https://finviz.com/screener.ashx?v=111&f=earningsdate_tomorrowbefore",
      "TomorrowAfter": "https://finviz.com/screener.ashx?v=111&f=earningsdate_tomorrowafter",
      "Yesterday": "https://finviz.com/screener.ashx?v=111&f=earningsdate_yesterday",
      "YesterdayBefore": "https://finviz.com/screener.ashx?v=111&f=earningsdate_yesterdaybefore",
      "YesterdayAfter": "https://finviz.com/screener.ashx?v=111&f=earningsdate_yesterdayafter",
      "Next5Days": "https://finviz.com/screener.ashx?v=111&f=earningsdate_nextdays5",
      "Previous5Days": "https://finviz.com/screener.ashx?v=111&f=earningsdate_prevdays5",
      "ThisWeek": "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek",
      "NextWeek": "https://finviz.com/screener.ashx?v=111&f=earningsdate_nextweek",
      "PreviousWeek": "https://finviz.com/screener.ashx?v=111&f=earningsdate_prevweek",
      "ThisMonth": "https://finviz.com/screener.ashx?v=111&f=earningsdate_thismonth"
    },
    "prices": {
      "Under1": "https://finviz.com/screener.ashx?v=111&f=sh_price_u1",
      "Under2": "https://finviz.com/screener.ashx?v=111&f=sh_price_u2",
      "Under3": "https://finviz.com/screener.ashx?v=111&f=sh_price_u3",
      "Under4": "https://finviz.com/screener.ashx?v=111&f=sh_price_u4",
      "Under5": "https://finviz.com/screener.ashx?v=111&f=sh_price_u5",
      "Under7": "https://finviz.com/screener.ashx?v=111&f=sh_price_u7",
      "Under10": "https://finviz.com/screener.ashx?v=111&f=sh_price_u10",
      "Under15": "https://finviz.com/screener.ashx?v=111&f=sh_price_u15",
      "Under20": "https://finviz.com/screener.ashx?v=111&f=sh_price_u20",
      "Under30": "https://finviz.com/screener.ashx?v=111&f=sh_price_u30",
      "Under40": "https://finviz.com/screener.ashx?v=111&f=sh_price_u40",
      "Under50": "https://finviz.com/screener.ashx?v=111&f=sh_price_u50",
      "Over1": "https://finviz.com/screener.ashx?v=111&f=sh_price_o1",
      "Over2": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2",
      "Over3": "https://finviz.com/screener.ashx?v=111&f=sh_price_o3",
      "Over4": "https://finviz.com/screener.ashx?v=111&f=sh_price_o4",
      "Over5": "https://finviz.com/screener.ashx?v=111&f=sh_price_o5",
      "Over7": "https://finviz.com/screener.ashx?v=111&f=sh_price_o7",
      "Over10": "https://finviz.com/screener.ashx?v=111&f=sh_price_o10",
      "Over15": "https://finviz.com/screener.ashx?v=111&f=sh_price_o15",
      "Over20": "https://finviz.com/screener.ashx?v=111&f=sh_price_o20",
      "Over30": "https://finviz.com/screener.ashx?v=111&f=sh_price_o30",
      "Over40": "https://finviz.com/screener.ashx?v=111&f=sh_price_o40",
      "Over50": "https://finviz.com/screener.ashx?v=111&f=sh_price_o50",
      "Over60": "https://finviz.com/screener.ashx?v=111&f=sh_price_o60",
      "Over70": "https://finviz.com/screener.ashx?v=111&f=sh_price_o70",
      "Over80": "https://finviz.com/screener.ashx?v=111&f=sh_price_o80",
      "Over90": "https://finviz.com/screener.ashx?v=111&f=sh_price_o90",
      "Over100": "https://finviz.com/screener.ashx?v=111&f=sh_price_o100",
      "1To5": "https://finviz.com/screener.ashx?v=111&f=sh_price_1to5",
      "1To10": "https://finviz.com/screener.ashx?v=111&f=sh_price_1to10",
      "1To20": "https://finviz.com/screener.ashx?v=111&f=sh_price_1to20",
      "5To10": "https://finviz.com/screener.ashx?v=111&f=sh_price_5to10",
      "5To20": "https://finviz.com/screener.ashx?v=111&f=sh_price_5to20",
      "5To50": "https://finviz.com/screener.ashx?v=111&f=sh_price_5to50",
      "10To20": "https://finviz.com/screener.ashx?v=111&f=sh_price_10to20",
      "10To50": "https://finviz.com/screener.ashx?v=111&f=sh_price_10to50",
      "20To50": "https://finviz.com/screener.ashx?v=111&f=sh_price_20to50",
      "50To100": "https://finviz.com/screener.ashx?v=111&f=sh_price_50to100"
    },
    "dividend_yield": {
      "None": "https://finviz.com/screener.ashx?v=111&f=fa_div_none",
      "Positive": "https://finviz.com/screener.ashx?v=111&f=fa_div_pos",
      "High": "https://finviz.com/screener.ashx?v=111&f=fa_div_high",
      "VeryHigh": "https://finviz.com/screener.ashx?v=111&f=fa_div_veryhigh",
      "Over1": "https://finviz.com/screener.ashx?v=111&f=fa_div_o1",
      "Over2": "https://finviz.com/screener.ashx?v=111&f=fa_div_o2",
      "Over3": "https://finviz.com/screener.ashx?v=111&f=fa_div_o3",
      "Over4": "https://finviz.com/screener.ashx?v=111&f=fa_div_o4",
      "Over5": "https://finviz.com/screener.ashx?v=111&f=fa_div_o5",
      "Over6": "https://finviz.com/screener.ashx?v=111&f=fa_div_o6",
      "Over7": "https://finviz.com/screener.ashx?v=111&f=fa_div_o7",
      "Over8": "https://finviz.com/screener.ashx?v=111&f=fa_div_o8",
      "Over9": "https://finviz.com/screener.ashx?v=111&f=fa_div_o9",
      "Over10": "https://finviz.com/screener.ashx?v=111&f=fa_div_o10"
    },
    "indices": {
      "SP500": "https://finviz.com/screener.ashx?v=111&f=idx_sp500",
      "Nasdaq100": "https://finviz.com/screener.ashx?v=111&f=idx_ndx",
      "DJIA": "https://finviz.com/screener.ashx?v=111&f=idx_dji",
      "Russell2000": "https://finviz.com/screener.ashx?v=111&f=idx_rut"
    },
    "average_volume": {
      "Under50K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_u50",
      "Under100K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_u100",
      "Under500K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_u500",
      "Under750K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_u750",
      "Under1M": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_u1000",
      "Over50K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o50",
      "Over100K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o100",
      "Over200K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o200",
      "Over300K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300",
      "Over400K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o400",
      "Over500K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500",
      "Over750K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o750",
      "Over1M": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o1000",
      "Over2M": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o2000",
      "100KTo500K": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_100to500",
      "100KTo1M": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_100to1000",
      "500KTo1M": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_500to1000",
      "500KTo10M": "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_500to10000"
    },
    "target_prices": {
      "Above50": "https://finviz.com/screener.ashx?v=111&f=targetprice_a50",
      "Above40": "https://finviz.com/screener.ashx?v=111&f=targetprice_a40",
      "Above30": "https://finviz.com/screener.ashx?v=111&f=targetprice_a30",
      "Above20": "https://finviz.com/screener.ashx?v=111&f=targetprice_a20",
      "Above10": "https://finviz.com/screener.ashx?v=111&f=targetprice_a10",
      "Above5": "https://finviz.com/screener.ashx?v=111&f=targetprice_a5",
      "AbovePrice": "https://finviz.com/screener.ashx?v=111&f=targetprice_above",
      "BelowPrice": "https://finviz.com/screener.ashx?v=111&f=targetprice_below",
      "Below5": "https://finviz.com/screener.ashx?v=111&f=targetprice_b5",
      "Below10": "https://finviz.com/screener.ashx?v=111&f=targetprice_b10",
      "Below20": "https://finviz.com/screener.ashx?v=111&f=targetprice_b20",
      "Below30": "https://finviz.com/screener.ashx?v=111&f=targetprice_b30",
      "Below40": "https://finviz.com/screener.ashx?v=111&f=targetprice_b40",
      "Below50": "https://finviz.com/screener.ashx?v=111&f=targetprice_b50"
    },
    "sectors": {
      "BasicMaterials": "https://finviz.com/screener.ashx?v=111&f=sec_basicmaterials",
      "CommunicationServices": "https://finviz.com/screener.ashx?v=111&f=sec_communicationservices",
      "ConsumerCyclical": "https://finviz.com/screener.ashx?v=111&f=sec_consumercyclical",
      "ConsumerDefensive": "https://finviz.com/screener.ashx?v=111&f=sec_consumerdefensive",
      "Energy": "https://finviz.com/screener.ashx?v=111&f=sec_energy",
      "Financial": "https://finviz.com/screener.ashx?v=111&f=sec_financial",
      "Healthcare": "https://finviz.com/screener.ashx?v=111&f=sec_healthcare",
      "Industrials": "https://finviz.com/screener.ashx?v=111&f=sec_industrials",
      "RealEstate": "https://finviz.com/screener.ashx?v=111&f=sec_realestate",
      "Technology": "https://finviz.com/screener.ashx?v=111&f=sec_technology",
      "Utilities": "https://finviz.com/screener.ashx?v=111&f=sec_utilities"
    },
    "float_short": {
      "Low": "https://finviz.com/screener.ashx?v=111&f=sh_short_low",
      "High": "https://finviz.com/screener.ashx?v=111&f=sh_short_high",
      "Under5": "https://finviz.com/screener.ashx?v=111&f=sh_short_u5",
      "Under10": "https://finviz.com/screener.ashx?v=111&f=sh_short_u10",
      "Under15": "https://finviz.com/screener.ashx?v=111&f=sh_short_u15",
      "Under20": "https://finviz.com/screener.ashx?v=111&f=sh_short_u20",
      "Under25": "https://finviz.com/screener.ashx?v=111&f=sh_short_u25",
      "Under30": "https://finviz.com/screener.ashx?v=111&f=sh_short_u30",
      "Over5": "https://finviz.com/screener.ashx?v=111&f=sh_short_o5",
      "Over10": "https://finviz.com/screener.ashx?v=111&f=sh_short_o10",
      "Over15": "https://finviz.com/screener.ashx?v=111&f=sh_short_o15",
      "Over20": "https://finviz.com/screener.ashx?v=111&f=sh_short_o20",
      "Over25": "https://finviz.com/screener.ashx?v=111&f=sh_short_o25",
      "Over30": "https://finviz.com/screener.ashx?v=111&f=sh_short_o30"
    },
    "relative_volume": {
      "Over10": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o10",
      "Over5": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o5",
      "Over3": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o3",
      "Over2": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o2",
      "Over1.5": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o1.5",
      "Over1": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o1",
      "Over0.75": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o0.75",
      "Over0.5": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o0.5",
      "Over0.25": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_o0.25",
      "Under2": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u2",
      "Under1.5": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u1.5",
      "Under1": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u1",
      "Under0.75": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u0.75",
      "Under0.5": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u0.5",
      "Under0.25": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u0.25",
      "Under0.1": "https://finviz.com/screener.ashx?v=111&f=sh_relvol_u0.1"
    },
    "ipo_date": {
      "Today": "https://finviz.com/screener.ashx?v=111&f=ipodate_today",
      "Yesterday": "https://finviz.com/screener.ashx?v=111&f=ipodate_yesterday",
      "PrevWeek": "https://finviz.com/screener.ashx?v=111&f=ipodate_prevweek",
      "PrevMonth": "https://finviz.com/screener.ashx?v=111&f=ipodate_prevmonth",
      "PrevQuarter": "https://finviz.com/screener.ashx?v=111&f=ipodate_prevquarter",
      "PrevYear": "https://finviz.com/screener.ashx?v=111&f=ipodate_prevyear",
      "Prev2Years": "https://finviz.com/screener.ashx?v=111&f=ipodate_prev2yrs",
      "Prev3Years": "https://finviz.com/screener.ashx?v=111&f=ipodate_prev3yrs",
      "Prev5Years": "https://finviz.com/screener.ashx?v=111&f=ipodate_prev5yrs",
      "More1": "https://finviz.com/screener.ashx?v=111&f=ipodate_more1",
      "More5": "https://finviz.com/screener.ashx?v=111&f=ipodate_more5",
      "More10": "https://finviz.com/screener.ashx?v=111&f=ipodate_more10",
      "More15": "https://finviz.com/screener.ashx?v=111&f=ipodate_more15",
      "More20": "https://finviz.com/screener.ashx?v=111&f=ipodate_more20",
      "More25": "https://finviz.com/screener.ashx?v=111&f=ipodate_more25"
    },
    "industries": {
      "StocksOnly": "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly",
      "ExchangeTradedFund": "https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund",
      "AdvertisingAgencies": "https://finviz.com/screener.ashx?v=111&f=ind_advertisingagencies",
      "AerospaceDefense": "https://finviz.com/screener.ashx?v=111&f=ind_aerospacedefense",
      "AgriculturalInputs": "https://finviz.com/screener.ashx?v=111&f=ind_agriculturalinputs",
      "Airlines": "https://finviz.com/screener.ashx?v=111&f=ind_airlines",
      "AirportsAirServices": "https://finviz.com/screener.ashx?v=111&f=ind_airportsairservices",
      "Aluminum": "https://finviz.com/screener.ashx?v=111&f=ind_aluminum",
      "ApparelManufacturing": "https://finviz.com/screener.ashx?v=111&f=ind_apparelmanufacturing",
      "ApparelRetail": "https://finviz.com/screener.ashx?v=111&f=ind_apparelretail",
      "AssetManagement": "https://finviz.com/screener.ashx?v=111&f=ind_assetm",
      "AutoManufacturers": "https://finviz.com/screener.ashx?v=111&f=ind_automanufacturers",
      "AutoParts": "https://finviz.com/screener.ashx?v=111&f=ind_autoparts",
      "AutoTruckDealerships": "https://finviz.com/screener.ashx?v=111&f=ind_autotruckdealerships",
      "BanksDiversified": "https://finviz.com/screener.ashx?v=111&f=ind_banksdiversified",
      "BanksRegional": "https://finviz.com/screener.ashx?v=111&f=ind_banksregional",
      "BeveragesBrewers": "https://finviz.com/screener.ashx?v=111&f=ind_beveragesbrewers",
      "BeveragesNonAlcoholic": "https://finviz.com/screener.ashx?v=111&f=ind_beveragesnonalcoholic",
      "BeveragesWineriesDistilleries": "https://finviz.com/screener.ashx?v=111&f=ind_beverageswineriesdistilleries",
      "Biotechnology": "https://finviz.com/screener.ashx?v=111&f=ind_biotechnology",
      "Broadcasting": "https://finviz.com/screener.ashx?v=111&f=ind_broadcasting",
      "BuildingMaterials": "https://finviz.com/screener.ashx?v=111&f=ind_buildingmaterials",
      "BuildingProductsEquipment": "https://finviz.com/screener.ashx?v=111&f=ind_buildingproductsequipment",
      "BusinessEquipmentSupplies": "https://finviz.com/screener.ashx?v=111&f=ind_businessequipmentsupplies",
      "CapitalMarkets": "https://finviz.com/screener.ashx?v=111&f=ind_capitalmarkets",
      "Chemicals": "https://finviz.com/screener.ashx?v=111&f=ind_chemicals",
      "ClosedEndFundDebt": "https://finviz.com/screener.ashx?v=111&f=ind_closedendfunddebt",
      "ClosedEndFundEquity": "https://finviz.com/screener.ashx?v=111&f=ind_closedendfundequity",
      "ClosedEndFundForeign": "https://finviz.com/screener.ashx?v=111&f=ind_closedendfundforeign",
      "CokingCoal": "https://finviz.com/screener.ashx?v=111&f=ind_cokingcoal",
      "CommunicationEquipment": "https://finviz.com/screener.ashx?v=111&f=ind_communicationequipment",
      "ComputerHardware": "https://finviz.com/screener.ashx?v=111&f=ind_computerhardware",
      "Confectioners": "https://finviz.com/screener.ashx?v=111&f=ind_confectioners",
      "Conglomerates": "https://finviz.com/screener.ashx?v=111&f=ind_conglomerates",
      "ConsultingServices": "https://finviz.com/screener.ashx?v=111&f=ind_consultingservices",
      "ConsumerElectronics": "https://finviz.com/screener.ashx?v=111&f=ind_consumerelectronics",
      "Copper": "https://finviz.com/screener.ashx?v=111&f=ind_copper",
      "CreditServices": "https://finviz.com/screener.ashx?v=111&f=ind_creditservices",
      "DepartmentStores": "https://finviz.com/screener.ashx?v=111&f=ind_departmentstores",
      "DiagnosticsResearch": "https://finviz.com/screener.ashx?v=111&f=ind_diagnosticsresearch",
      "DiscountStores": "https://finviz.com/screener.ashx?v=111&f=ind_discountstores",
      "DrugManufacturersGeneral": "https://finviz.com/screener.ashx?v=111&f=ind_drugmanufacturersgeneral",
      "DrugManufacturersSpecialtyGeneric": "https://finviz.com/screener.ashx?v=111&f=ind_drugmanufacturersspecialtygeneric",
      "EducationTrainingServices": "https://finviz.com/screener.ashx?v=111&f=ind_educationtrainingservices",
      "ElectricalEquipmentParts": "https://finviz.com/screener.ashx?v=111&f=ind_electricalequipmentparts",
      "ElectronicComponents": "https://finviz.com/screener.ashx?v=111&f=ind_electroniccomponents",
      "ElectronicGamingMultimedia": "https://finviz.com/screener.ashx?v=111&f=ind_electronicgamingmultimedia",
      "ElectronicsComputerDistribution": "https://finviz.com/screener.ashx?v=111&f=ind_electronicscomputerdistribution",
      "EngineeringConstruction": "https://finviz.com/screener.ashx?v=111&f=ind_engineeringconstruction",
      "Entertainment": "https://finviz.com/screener.ashx?v=111&f=ind_entertainment",
      "FarmHeavyConstructionMachinery": "https://finviz.com/screener.ashx?v=111&f=ind_farmheavyconstructionmachinery",
      "FarmProducts": "https://finviz.com/screener.ashx?v=111&f=ind_farmproducts",
      "FinancialConglomerates": "https://finviz.com/screener.ashx?v=111&f=ind_financialconglomerates",
      "FinancialDataStockExchanges": "https://finviz.com/screener.ashx?v=111&f=ind_financialdatastockexchanges",
      "FoodDistribution": "https://finviz.com/screener.ashx?v=111&f=ind_fooddistribution",
      "FootwearAccessories": "https://finviz.com/screener.ashx?v=111&f=ind_footwearaccessories",
      "FurnishingsFixturesAppliances": "https://finviz.com/screener.ashx?v=111&f=ind_furnishingsfixturesappliances",
      "Gambling": "https://finviz.com/screener.ashx?v=111&f=ind_gambling",
      "Gold": "https://finviz.com/screener.ashx?v=111&f=ind_gold",
      "GroceryStores": "https://finviz.com/screener.ashx?v=111&f=ind_grocerystores",
      "HealthcarePlans": "https://finviz.com/screener.ashx?v=111&f=ind_healthcareplans",
      "HealthInformationServices": "https://finviz.com/screener.ashx?v=111&f=ind_healthinformationservices",
      "HomeImprovementRetail": "https://finviz.com/screener.ashx?v=111&f=ind_homeimprovementretail",
      "HouseholdPersonalProducts": "https://finviz.com/screener.ashx?v=111&f=ind_householdpersonalproducts",
      "IndustrialDistribution": "https://finviz.com/screener.ashx?v=111&f=ind_industrialdistribution",
      "InformationTechnologyServices": "https://finviz.com/screener.ashx?v=111&f=ind_informationtechnologyservices",
      "InfrastructureOperations": "https://finviz.com/screener.ashx?v=111&f=ind_infrastructureoperations",
      "InsuranceBrokers": "https://finviz.com/screener.ashx?v=111&f=ind_insurancebrokers",
      "InsuranceDiversified": "https://finviz.com/screener.ashx?v=111&f=ind_insurancediversified",
      "InsuranceLife": "https://finviz.com/screener.ashx?v=111&f=ind_insurancelife",
      "InsurancePropertyCasualty": "https://finviz.com/screener.ashx?v=111&f=ind_insurancepropertycasualty",
      "InsuranceReinsurance": "https://finviz.com/screener.ashx?v=111&f=ind_insurancereinsurance",
      "InsuranceSpecialty": "https://finviz.com/screener.ashx?v=111&f=ind_insurancespecialty",
      "IntegratedFreightLogistics": "https://finviz.com/screener.ashx?v=111&f=ind_integratedfreightlogistics",
      "InternetContentInformation": "https://finviz.com/screener.ashx?v=111&f=ind_internetcontentinformation",
      "InternetRetail": "https://finviz.com/screener.ashx?v=111&f=ind_internetretail",
      "Leisure": "https://finviz.com/screener.ashx?v=111&f=ind_leisure",
      "Lodging": "https://finviz.com/screener.ashx?v=111&f=ind_lodging",
      "LumberWoodProduction": "https://finviz.com/screener.ashx?v=111&f=ind_lumberwoodproduction",
      "LuxuryGoods": "https://finviz.com/screener.ashx?v=111&f=ind_luxurygoods",
      "MarineShipping": "https://finviz.com/screener.ashx?v=111&f=ind_marineshipping",
      "MedicalCareFacilities": "https://finviz.com/screener.ashx?v=111&f=ind_medicalcarefacilities",
      "MedicalDevices": "https://finviz.com/screener.ashx?v=111&f=ind_medicaldevices",
      "MedicalDistribution": "https://finviz.com/screener.ashx?v=111&f=ind_medicaldistribution",
      "MedicalInstrumentsSupplies": "https://finviz.com/screener.ashx?v=111&f=ind_medicalinstrumentssupplies",
      "MetalFabrication": "https://finviz.com/screener.ashx?v=111&f=ind_metalfabrication",
      "MortgageFinance": "https://finviz.com/screener.ashx?v=111&f=ind_mortgagefinance",
      "OilGasDrilling": "https://finviz.com/screener.ashx?v=111&f=ind_oilgasdrilling",
      "OilGasEP": "https://finviz.com/screener.ashx?v=111&f=ind_oilgasep",
      "OilGasEquipmentServices": "https://finviz.com/screener.ashx?v=111&f=ind_oilgasequipmentservices",
      "OilGasIntegrated": "https://finviz.com/screener.ashx?v=111&f=ind_oilgasintegrated",
      "OilGasMidstream": "https://finviz.com/screener.ashx?v=111&f=ind_oilgasmidstream",
      "OilGasRefiningMarketing": "https://finviz.com/screener.ashx?v=111&f=ind_oilgasmidstream",
      "OtherIndustrialMetalsMining": "https://finviz.com/screener.ashx?v=111&f=ind_otherindustrialmetalsmining",
      "PackagedFoods": "https://finviz.com/screener.ashx?v=111&f=ind_packagedfoods",
      "PackagingContainers": "https://finviz.com/screener.ashx?v=111&f=ind_packagingcontainers",
      "PaperPaperProducts": "https://finviz.com/screener.ashx?v=111&f=ind_paperpaperproducts",
      "PersonalServices": "https://finviz.com/screener.ashx?v=111&f=ind_personalservices",
      "PharmaceuticalRetailers": "https://finviz.com/screener.ashx?v=111&f=ind_pharmaceuticalretailers",
      "PollutionTreatmentControls": "https://finviz.com/screener.ashx?v=111&f=ind_pollutiontreatmentcontrols",
      "Publishing": "https://finviz.com/screener.ashx?v=111&f=ind_publishing",
      "Railroads": "https://finviz.com/screener.ashx?v=111&f=ind_railroads",
      "RealEstateDevelopment": "https://finviz.com/screener.ashx?v=111&f=ind_realestatedevelopment",
      "RealEstateDiversified": "https://finviz.com/screener.ashx?v=111&f=ind_realestatediversified",
      "RealEstateServices": "https://finviz.com/screener.ashx?v=111&f=ind_realestateservices",
      "RecreationalVehicles": "https://finviz.com/screener.ashx?v=111&f=ind_recreationalvehicles",
      "REITDiversified": "https://finviz.com/screener.ashx?v=111&f=ind_reitdiversified",
      "REITHealthcareFacilities": "https://finviz.com/screener.ashx?v=111&f=ind_reithealthcarefacilities",
      "REITHotelMotel": "https://finviz.com/screener.ashx?v=111&f=ind_reithotelmotel",
      "REITIndustrial": "https://finviz.com/screener.ashx?v=111&f=ind_reitindustrial",
      "REITMortgage": "https://finviz.com/screener.ashx?v=111&f=ind_reitmortgage",
      "REITOffice": "https://finviz.com/screener.ashx?v=111&f=ind_reitoffice",
      "REITResidential": "https://finviz.com/screener.ashx?v=111&f=ind_reitresidential",
      "REITRetail": "https://finviz.com/screener.ashx?v=111&f=ind_reitretail",
      "REITSpecialty": "https://finviz.com/screener.ashx?v=111&f=ind_reitspecialty",
      "RentalLeasingServices": "https://finviz.com/screener.ashx?v=111&f=ind_rentalleasingservices",
      "ResidentialConstruction": "https://finviz.com/screener.ashx?v=111&f=ind_residentialconstruction",
      "ResortsCasinos": "https://finviz.com/screener.ashx?v=111&f=ind_resortscasinos",
      "Restaurants": "https://finviz.com/screener.ashx?v=111&f=ind_restaurants",
      "ScientificTechnicalInstruments": "https://finviz.com/screener.ashx?v=111&f=ind_scientifictechnicalinstruments",
      "SecurityProtectionServices": "https://finviz.com/screener.ashx?v=111&f=ind_securityprotectionservices",
      "SemiconductorEquipmentMaterials": "https://finviz.com/screener.ashx?v=111&f=ind_semiconductorequipmentmaterials",
      "Semiconductors": "https://finviz.com/screener.ashx?v=111&f=ind_semiconductors",
      "ShellCompanies": "https://finviz.com/screener.ashx?v=111&f=ind_shellcompanies",
      "Silver": "https://finviz.com/screener.ashx?v=111&f=ind_silver",
      "SoftwareApplication": "https://finviz.com/screener.ashx?v=111&f=ind_softwareapplication",
      "SoftwareInfrastructure": "https://finviz.com/screener.ashx?v=111&f=ind_softwareinfrastructure",
      "Solar": "https://finviz.com/screener.ashx?v=111&f=ind_solar",
      "SpecialtyBusinessServices": "https://finviz.com/screener.ashx?v=111&f=ind_specialtybusinessservices",
      "SpecialtyChemicals": "https://finviz.com/screener.ashx?v=111&f=ind_specialtychemicals",
      "SpecialtyIndustrialMachinery": "https://finviz.com/screener.ashx?v=111&f=ind_specialtyindustrialmachinery",
      "SpecialtyRetail": "https://finviz.com/screener.ashx?v=111&f=ind_specialtyretail",
      "StaffingEmploymentServices": "https://finviz.com/screener.ashx?v=111&f=ind_staffingemploymentservices",
      "Steel": "https://finviz.com/screener.ashx?v=111&f=ind_steel",
      "TelecomServices": "https://finviz.com/screener.ashx?v=111&f=ind_telecomservices",
      "TextileManufacturing": "https://finviz.com/screener.ashx?v=111&f=ind_textilemanufacturing",
      "ThermalCoal": "https://finviz.com/screener.ashx?v=111&f=ind_thermalcoal",
      "Tobacco": "https://finviz.com/screener.ashx?v=111&f=ind_tobacco",
      "ToolsAccessories": "https://finviz.com/screener.ashx?v=111&f=ind_toolsaccessories",
      "TravelServices": "https://finviz.com/screener.ashx?v=111&f=ind_travelservices",
      "Trucking": "https://finviz.com/screener.ashx?v=111&f=ind_trucking",
      "Uranium": "https://finviz.com/screener.ashx?v=111&f=ind_uranium",
      "UtilitiesDiversified": "https://finviz.com/screener.ashx?v=111&f=ind_utilitiesdiversified",
      "UtilitiesIndependentPowerProducers": "https://finviz.com/screener.ashx?v=111&f=ind_utilitiesindependentpowerproducers",
      "UtilitiesRegulatedElectric": "https://finviz.com/screener.ashx?v=111&f=ind_utilitiesregulatedelectric",
      "UtilitiesRegulatedGas": "https://finviz.com/screener.ashx?v=111&f=ind_utilitiesregulatedgas",
      "UtilitiesRegulatedWater": "https://finviz.com/screener.ashx?v=111&f=ind_utilitiesregulatedwater",
      "UtilitiesRenewable": "https://finviz.com/screener.ashx?v=111&f=ind_utilitiesrenewable",
      "WasteManagement": "https://finviz.com/screener.ashx?v=111&f=ind_wastemanagement"
    },
    "analyst_recommendations": {
      "StrongBuy": "https://finviz.com/screener.ashx?v=111&f=an_recom_strongbuy",
      "BuyBetter": "https://finviz.com/screener.ashx?v=111&f=an_recom_buybetter",
      "Buy": "https://finviz.com/screener.ashx?v=111&f=an_recom_buy",
      "HoldBetter": "https://finviz.com/screener.ashx?v=111&f=an_recom_holdbetter",
      "Hold": "https://finviz.com/screener.ashx?v=111&f=an_recom_hold",
      "HoldWorse": "https://finviz.com/screener.ashx?v=111&f=an_recom_holdworse",
      "Sell": "https://finviz.com/screener.ashx?v=111&f=an_recom_sell",
      "SellWorse": "https://finviz.com/screener.ashx?v=111&f=an_recom_sellworse",
      "StrongSell": "https://finviz.com/screener.ashx?v=111&f=an_recom_strongsell"
    },
    "current_volume": {
      "Under50K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_u50",
      "Under100K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_u100",
      "Under500K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_u500",
      "Under750K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_u750",
      "Under1M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_u1000",
      "Over0": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o0",
      "Over50K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o50",
      "Over100K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o100",
      "Over200K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o200",
      "Over300K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o300",
      "Over400K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o400",
      "Over500K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o500",
      "Over750K": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o750",
      "Over1M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o1000",
      "Over2M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o2000",
      "Over5M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o5000",
      "Over10M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o10000",
      "Over20M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o20000",
      "Over50PercentSharesFloat": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o50sf",
      "Over100PercentSharesFloat": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_o100sf"
    },
    "current_volume_usd": {
      "Under1M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_uusd1000",
      "Under10M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_uusd10000",
      "Under100M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_uusd100000",
      "Under1B": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_uusd1000000",
      "Over1M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_ousd1000",
      "Over10M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_ousd10000",
      "Over100M": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_ousd100000",
      "Over1B": "https://finviz.com/screener.ashx?v=111&f=sh_curvol_ousd1000000"
    },
    "shares_outstanding": {
      "Under1M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_u1",
      "Under5M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_u5",
      "Under10M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_u10",
      "Under20M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_u20",
      "Under50M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_u50",
      "Under100M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_u100",
      "Over1M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o1",
      "Over2M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o2",
      "Over5M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o5",
      "Over10M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o10",
      "Over20M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o20",
      "Over50M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o50",
      "Over100M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o100",
      "Over200M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o200",
      "Over500M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o500",
      "Over1000M": "https://finviz.com/screener.ashx?v=111&f=sh_outstanding_o1000"
    },
  "Countries": {
    "USA": "https://finviz.com/screener.ashx?v=111&f=geo_usa&ft=4",
    "Foreign (ex-USA)": "https://finviz.com/screener.ashx?v=111&f=geo_notusa",
    "Asia": "https://finviz.com/screener.ashx?v=111&f=geo_asia",
    "Europe": "https://finviz.com/screener.ashx?v=111&f=geo_europe",
    "Latin America": "https://finviz.com/screener.ashx?v=111&f=geo_latinamerica",
    "BRIC": "https://finviz.com/screener.ashx?v=111&f=geo_bric",
    "Argentina": "https://finviz.com/screener.ashx?v=111&f=geo_argentina",
    "Australia": "https://finviz.com/screener.ashx?v=111&f=geo_australia",
    "Bahamas": "https://finviz.com/screener.ashx?v=111&f=geo_bahamas",
    "Belgium": "https://finviz.com/screener.ashx?v=111&f=geo_belgium",
    "BeNeLux": "https://finviz.com/screener.ashx?v=111&f=geo_benelux",
    "Bermuda": "https://finviz.com/screener.ashx?v=111&f=geo_bermuda",
    "Brazil": "https://finviz.com/screener.ashx?v=111&f=geo_brazil",
    "Canada": "https://finviz.com/screener.ashx?v=111&f=geo_canada",
    "Cayman Islands": "https://finviz.com/screener.ashx?v=111&f=geo_caymanislands",
    "Chile": "https://finviz.com/screener.ashx?v=111&f=geo_chile",
    "China": "https://finviz.com/screener.ashx?v=111&f=geo_china",
    "China & Hong Kong": "https://finviz.com/screener.ashx?v=111&f=geo_chinahongkong",
    "Colombia": "https://finviz.com/screener.ashx?v=111&f=geo_colombia",
    "Cyprus": "https://finviz.com/screener.ashx?v=111&f=geo_cyprus",
    "Denmark": "https://finviz.com/screener.ashx?v=111&f=geo_denmark",
    "Finland": "https://finviz.com/screener.ashx?v=111&f=geo_finland",
    "France": "https://finviz.com/screener.ashx?v=111&f=geo_france",
    "Germany": "https://finviz.com/screener.ashx?v=111&f=geo_germany",
    "Greece": "https://finviz.com/screener.ashx?v=111&f=geo_greece",
    "Hong Kong": "https://finviz.com/screener.ashx?v=111&f=geo_hongkong",
    "Hungary": "https://finviz.com/screener.ashx?v=111&f=geo_hungary",
    "Iceland": "https://finviz.com/screener.ashx?v=111&f=geo_iceland",
    "India": "https://finviz.com/screener.ashx?v=111&f=geo_india",
    "Indonesia": "https://finviz.com/screener.ashx?v=111&f=geo_indonesia",
    "Ireland": "https://finviz.com/screener.ashx?v=111&f=geo_ireland",
    "Israel": "https://finviz.com/screener.ashx?v=111&f=geo_israel",
    "Italy": "https://finviz.com/screener.ashx?v=111&f=geo_italy",
    "Japan": "https://finviz.com/screener.ashx?v=111&f=geo_japan",
    "Kazakhstan": "https://finviz.com/screener.ashx?v=111&f=geo_kazakhstan",
    "Luxembourg": "https://finviz.com/screener.ashx?v=111&f=geo_luxembourg",
    "Malaysia": "https://finviz.com/screener.ashx?v=111&f=geo_malaysia",
    "Malta": "https://finviz.com/screener.ashx?v=111&f=geo_malta",
    "Mexico": "https://finviz.com/screener.ashx?v=111&f=geo_mexico",
    "Monaco": "https://finviz.com/screener.ashx?v=111&f=geo_monaco",
    "Netherlands": "https://finviz.com/screener.ashx?v=111&f=geo_netherlands",
    "New Zealand": "https://finviz.com/screener.ashx?v=111&f=geo_newzealand",
    "Norway": "https://finviz.com/screener.ashx?v=111&f=geo_norway",
    "Panama": "https://finviz.com/screener.ashx?v=111&f=geo_panama",
    "Peru": "https://finviz.com/screener.ashx?v=111&f=geo_peru",
    "Philippines": "https://finviz.com/screener.ashx?v=111&f=geo_philippines",
    "Portugal": "https://finviz.com/screener.ashx?v=111&f=geo_portugal",
    "Russia": "https://finviz.com/screener.ashx?v=111&f=geo_russia",
    "Singapore": "https://finviz.com/screener.ashx?v=111&f=geo_singapore",
    "South Africa": "https://finviz.com/screener.ashx?v=111&f=geo_southafrica",
    "South Korea": "https://finviz.com/screener.ashx?v=111&f=geo_southkorea",
    "Spain": "https://finviz.com/screener.ashx?v=111&f=geo_spain",
    "Sweden": "https://finviz.com/screener.ashx?v=111&f=geo_sweden",
    "Switzerland": "https://finviz.com/screener.ashx?v=111&f=geo_switzerland",
    "Taiwan": "https://finviz.com/screener.ashx?v=111&f=geo_taiwan",
    "Turkey": "https://finviz.com/screener.ashx?v=111&f=geo_turkey",
    "United Arab Emirates": "https://finviz.com/screener.ashx?v=111&f=geo_unitedarabemirates",
    "United Kingdom": "https://finviz.com/screener.ashx?v=111&f=geo_unitedkingdom",
    "Uruguay": "https://finviz.com/screener.ashx?v=111&f=geo_uruguay"
  },
  "Float": {
    "Under 1M": "https://finviz.com/screener.ashx?v=111&f=sh_float_u1",
    "Under 5M": "https://finviz.com/screener.ashx?v=111&f=sh_float_u5",
    "Under 10M": "https://finviz.com/screener.ashx?v=111&f=sh_float_u10",
    "Under 20M": "https://finviz.com/screener.ashx?v=111&f=sh_float_u20",
    "Under 50M": "https://finviz.com/screener.ashx?v=111&f=sh_float_u50",
    "Under 100M": "https://finviz.com/screener.ashx?v=111&f=sh_float_u100",
    "Over 1M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o1",
    "Over 2M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o2",
    "Over 5M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o5",
    "Over 10M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o10",
    "Over 20M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o20",
    "Over 50M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o50",
    "Over 100M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o100",
    "Over 200M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o200",
    "Over 500M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o500",
    "Over 1000M": "https://finviz.com/screener.ashx?v=111&f=sh_float_o1000",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u10p",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u20p",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u30p",
    "Under 40%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u40p",
    "Under 50%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u50p",
    "Under 60%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u60p",
    "Under 70%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u70p",
    "Under 80%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u80p",
    "Under 90%": "https://finviz.com/screener.ashx?v=111&f=sh_float_u90p",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o10p",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o20p",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o30p",
    "Over 40%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o40p",
    "Over 50%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o50p",
    "Over 60%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o60p",
    "Over 70%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o70p",
    "Over 80%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o80p",
    "Over 90%": "https://finviz.com/screener.ashx?v=111&f=sh_float_o90p"
  },
  "P/B Ratio": {
    "Low (<1)": "https://finviz.com/screener.ashx?v=111&f=fa_pb_low&ft=4",
    "High (>5)": "https://finviz.com/screener.ashx?v=111&f=fa_pb_high&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u1&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u2&ft=4",
    "Under 3": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u3&ft=4",
    "Under 4": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u4&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u5&ft=4",
    "Under 6": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u6&ft=4",
    "Under 7": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u7&ft=4",
    "Under 8": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u8&ft=4",
    "Under 9": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u9&ft=4",
    "Under 10": "https://finviz.com/screener.ashx?v=111&f=fa_pb_u10&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o1&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o2&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o3&ft=4",
    "Over 4": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o4&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o5&ft=4",
    "Over 6": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o6&ft=4",
    "Over 7": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o7&ft=4",
    "Over 8": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o8&ft=4",
    "Over 9": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o9&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_pb_o10&ft=4"
  },
  "EPS Growth (Past 5 Years)": {
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_neg&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_pos&ft=4",
    "Positive Low (0-10%)": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_poslow&ft=4",
    "High (>25%)": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_high&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_u5&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_u10&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_u15&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_u20&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_u25&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_u30&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_eps5years_o30&ft=4"
  },
  "Earnings Surprise": {
    "Both Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_bp&ft=4",
    "Both Met (0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_bm&ft=4",
    "Both Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_bn&ft=4",
    "EPS Surprise: Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ep&ft=4",
    "EPS Surprise: Met (0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_em&ft=4",
    "EPS Surprise: Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_en&ft=4",
    "EPS Surprise: Under -100%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu100&ft=4",
    "EPS Surprise: Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu50&ft=4",
    "EPS Surprise: Under -40%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu40&ft=4",
    "EPS Surprise: Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu30&ft=4",
    "EPS Surprise: Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu20&ft=4",
    "EPS Surprise: Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu10&ft=4",
    "EPS Surprise: Under -5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eu5&ft=4",
    "EPS Surprise: Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo5&ft=4",
    "EPS Surprise: Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo10&ft=4",
    "EPS Surprise: Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo20&ft=4",
    "EPS Surprise: Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo30&ft=4",
    "EPS Surprise: Over 40%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo40&ft=4",
    "EPS Surprise: Over 50%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo50&ft=4",
    "EPS Surprise: Over 60%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo60&ft=4",
    "EPS Surprise: Over 70%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo70&ft=4",
    "EPS Surprise: Over 80%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo80&ft=4",
    "EPS Surprise: Over 90%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo90&ft=4",
    "EPS Surprise: Over 100%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo100&ft=4",
    "EPS Surprise: Over 200%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_eo200&ft=4",
    "Revenue Surprise: Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_rp&ft=4",
    "Revenue Surprise: Met (0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_rm&ft=4",
    "Revenue Surprise: Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_rn&ft=4",
    "Revenue Surprise: Under -100%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru100&ft=4",
    "Revenue Surprise: Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru50&ft=4",
    "Revenue Surprise: Under -40%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru40&ft=4",
    "Revenue Surprise: Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru30&ft=4",
    "Revenue Surprise: Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru20&ft=4",
    "Revenue Surprise: Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru10&ft=4",
    "Revenue Surprise: Under -5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ru5&ft=4",
    "Revenue Surprise: Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro5&ft=4",
    "Revenue Surprise: Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro10&ft=4",
    "Revenue Surprise: Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro20&ft=4",
    "Revenue Surprise: Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro30&ft=4",
    "Revenue Surprise: Over 40%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro40&ft=4",
    "Revenue Surprise: Over 50%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro50&ft=4",
    "Revenue Surprise: Over 60%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro60&ft=4",
    "Revenue Surprise: Over 70%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro70&ft=4",
    "Revenue Surprise: Over 80%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro80&ft=4",
    "Revenue Surprise: Over 90%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro90&ft=4",
    "Revenue Surprise: Over 100%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro100&ft=4",
    "Revenue Surprise: Over 200%": "https://finviz.com/screener.ashx?v=111&f=fa_epsrev_ro200&ft=4"
  },
  "Current Ratio": {
    "High (>3)": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_high&ft=4",
    "Low (<1)": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_low&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_u1&ft=4",
    "Under 1.5": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_u15&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_u2&ft=4",
    "Under 2.5": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_u25&ft=4",
    "Under 3": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_u3&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_o1&ft=4",
    "Over 1.5": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_o15&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_o2&ft=4",
    "Over 2.5": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_o25&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=fa_curratio_o3&ft=4"
  },
 "Institutional Ownership": {
    "Low (<5%)": "https://finviz.com/screener.ashx?v=111&f=sh_instown_low&ft=4",
    "High (>90%)": "https://finviz.com/screener.ashx?v=111&f=sh_instown_high&ft=4",
    "Under 90%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u90&ft=4",
    "Under 80%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u80&ft=4",
    "Under 70%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u70&ft=4",
    "Under 60%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u60&ft=4",
    "Under 50%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u50&ft=4",
    "Under 40%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u40&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u30&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u20&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_u10&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o10&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o20&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o30&ft=4",
    "Over 40%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o40&ft=4",
    "Over 50%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o50&ft=4",
    "Over 60%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o60&ft=4",
    "Over 70%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o70&ft=4",
    "Over 80%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o80&ft=4",
    "Over 90%": "https://finviz.com/screener.ashx?v=111&f=sh_instown_o90&ft=4"
  },
  "Gap": {
    "Up": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u&ft=4",
    "Up 0%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u0&ft=4",
    "Up 1%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u1&ft=4",
    "Up 2%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u2&ft=4",
    "Up 3%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u3&ft=4",
    "Up 4%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u4&ft=4",
    "Up 5%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u5&ft=4",
    "Up 6%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u6&ft=4",
    "Up 7%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u7&ft=4",
    "Up 8%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u8&ft=4",
    "Up 9%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u9&ft=4",
    "Up 10%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u10&ft=4",
    "Up 15%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u15&ft=4",
    "Up 20%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_u20&ft=4",
    "Down": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d&ft=4",
    "Down 0%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d0&ft=4",
    "Down 1%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d1&ft=4",
    "Down 2%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d2&ft=4",
    "Down 3%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d3&ft=4",
    "Down 4%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d4&ft=4",
    "Down 5%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d5&ft=4",
    "Down 6%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d6&ft=4",
    "Down 7%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d7&ft=4",
    "Down 8%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d8&ft=4",
    "Down 9%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d9&ft=4",
    "Down 10%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d10&ft=4",
    "Down 15%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d15&ft=4",
    "Down 20%": "https://finviz.com/screener.ashx?v=111&f=ta_gap_d20&ft=4"
  },
  "Pattern": {
    "Horizontal S/R": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_horizontal&ft=4",
    "Horizontal S/R (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_horizontal2&ft=4",
    "TL Resistance": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_tlresistance&ft=4",
    "TL Resistance (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_tlresistance2&ft=4",
    "TL Support": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_tlsupport&ft=4",
    "TL Support (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_tlsupport2&ft=4",
    "Wedge Up": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgeup&ft=4",
    "Wedge Up (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgeup2&ft=4",
    "Wedge Down": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgedown&ft=4",
    "Wedge Down (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgedown2&ft=4",
    "Triangle Ascending": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgeresistance&ft=4",
    "Triangle Ascending (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgeresistance2&ft=4",
    "Triangle Descending": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgesupport&ft=4",
    "Triangle Descending (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedgesupport2&ft=4",
    "Wedge": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedge&ft=4",
    "Wedge (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_wedge2&ft=4",
    "Channel Up": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_channelup&ft=4",
    "Channel Up (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_channelup2&ft=4",
    "Channel Down": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_channeldown&ft=4",
    "Channel Down (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_channeldown2&ft=4",
    "Channel": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_channel&ft=4",
    "Channel (Strong)": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_channel2&ft=4",
    "Double Top": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_doubletop&ft=4",
    "Double Bottom": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_doublebottom&ft=4",
    "Multiple Top": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_multipletop&ft=4",
    "Multiple Bottom": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_multiplebottom&ft=4",
    "Head & Shoulders": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_headandshoulders&ft=4",
    "Head & Shoulders Inverse": "https://finviz.com/screener.ashx?v=111&f=ta_pattern_headandshouldersinv&ft=4"
  },
  "P/E": {
    "Low (<15)": "https://finviz.com/screener.ashx?v=111&f=fa_pe_low&ft=4",
    "Profitable (>0)": "https://finviz.com/screener.ashx?v=111&f=fa_pe_profitable&ft=4",
    "High (>50)": "https://finviz.com/screener.ashx?v=111&f=fa_pe_high&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u5&ft=4",
    "Under 10": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u10&ft=4",
    "Under 15": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u15&ft=4",
    "Under 20": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u20&ft=4",
    "Under 25": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u25&ft=4",
    "Under 30": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u30&ft=4",
    "Under 35": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u35&ft=4",
    "Under 40": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u40&ft=4",
    "Under 45": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u45&ft=4",
    "Under 50": "https://finviz.com/screener.ashx?v=111&f=fa_pe_u50&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o5&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o10&ft=4",
    "Over 15": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o15&ft=4",
    "Over 20": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o20&ft=4",
    "Over 25": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o25&ft=4",
    "Over 30": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o30&ft=4",
    "Over 35": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o35&ft=4",
    "Over 40": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o40&ft=4",
    "Over 45": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o45&ft=4",
    "Over 50": "https://finviz.com/screener.ashx?v=111&f=fa_pe_o50&ft=4"
  },
  "Price/Cash": {
    "Low (<3)": "https://finviz.com/screener.ashx?v=111&f=fa_pc_low&ft=4",
    "High (>50)": "https://finviz.com/screener.ashx?v=111&f=fa_pc_high&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u1&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u2&ft=4",
    "Under 3": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u3&ft=4",
    "Under 4": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u4&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u5&ft=4",
    "Under 6": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u6&ft=4",
    "Under 7": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u7&ft=4",
    "Under 8": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u8&ft=4",
    "Under 9": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u9&ft=4",
    "Under 10": "https://finviz.com/screener.ashx?v=111&f=fa_pc_u10&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o1&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o2&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o3&ft=4",
    "Over 4": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o4&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o5&ft=4",
    "Over 6": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o6&ft=4",
    "Over 7": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o7&ft=4",
    "Over 8": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o8&ft=4",
    "Over 9": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o9&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o10&ft=4",
    "Over 20": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o20&ft=4",
    "Over 30": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o30&ft=4",
    "Over 40": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o40&ft=4",
    "Over 50": "https://finviz.com/screener.ashx?v=111&f=fa_pc_o50&ft=4"
  },
  "EPS Growth (Next 5 Years)": {
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_neg&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_pos&ft=4",
    "Positive Low (<10%)": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_poslow&ft=4",
    "High (>25%)": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_high&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_u5&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_u10&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_u15&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_u20&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_u25&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_u30&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_estltgrowth_o30&ft=4"
  },
  "Quick Ratio": {
    "High (>3)": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_high&ft=4",
    "Low (<0.5)": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_low&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_u1&ft=4",
    "Under 0.5": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_u0.5&ft=4",
    "Over 0.5": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o0.5&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o1&ft=4",
    "Over 1.5": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o1.5&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o2&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o3&ft=4",
    "Over 4": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o4&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o5&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_quickratio_o10&ft=4"
  },
  "Net Profit Margin": {
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_pos&ft=4",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_neg&ft=4",
    "Very Negative (<=-20%)": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_veryneg&ft=4",
    "High (>20%)": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_high&ft=4",
    "Under 90%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u90&ft=4",
    "Under 80%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u80&ft=4",
    "Under 70%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u70&ft=4",
    "Under 60%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u60&ft=4",
    "Under 50%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u50&ft=4",
    "Under 45%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u45&ft=4",
    "Under 40%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u40&ft=4",
    "Under 35%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u35&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u30&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u25&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u20&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u15&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u10&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u5&ft=4",
    "Under 0%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u0&ft=4",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u-10&ft=4",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u-20&ft=4",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u-30&ft=4",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u-50&ft=4",
    "Under -70%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u-70&ft=4",
    "Under -100%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_u-100&ft=4",
    "Over 0%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o0&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o30&ft=4",
    "Over 35%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o35&ft=4",
    "Over 40%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o40&ft=4",
    "Over 45%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o45&ft=4",
    "Over 50%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o50&ft=4",
    "Over 60%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o60&ft=4",
    "Over 70%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o70&ft=4",
    "Over 80%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o80&ft=4",
    "Over 90%": "https://finviz.com/screener.ashx?v=111&f=fa_netmargin_o90&ft=4"
  },
  "Institutional Transactions": {
    "Very Negative (<-20%)": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_veryneg&ft=4",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_neg&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_pos&ft=4",
    "Very Positive (>20%)": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_verypos&ft=4",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-50&ft=4",
    "Under -45%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-45&ft=4",
    "Under -40%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-40&ft=4",
    "Under -35%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-35&ft=4",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-30&ft=4",
    "Under -25%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-25&ft=4",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-20&ft=4",
    "Under -15%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-15&ft=4",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-10&ft=4",
    "Under -5%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_u-5&ft=4",
    "Over +5%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o5&ft=4",
    "Over +10%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o10&ft=4",
    "Over +15%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o15&ft=4",
    "Over +20%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o20&ft=4",
    "Over +25%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o25&ft=4",
    "Over +30%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o30&ft=4",
    "Over +35%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o35&ft=4",
    "Over +40%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o40&ft=4",
    "Over +45%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o45&ft=4",
    "Over +50%": "https://finviz.com/screener.ashx?v=111&f=sh_insttrans_o50&ft=4"
  },
  "Price/SMA20": {
    "Price below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pb&ft=4",
    "Price 10% below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pb10&ft=4",
    "Price 20% below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pb20&ft=4",
    "Price 30% below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pb30&ft=4",
    "Price 40% below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pb40&ft=4",
    "Price 50% below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pb50&ft=4",
    "Price above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pa&ft=4",
    "Price 10% above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pa10&ft=4",
    "Price 20% above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pa20&ft=4",
    "Price 30% above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pa30&ft=4",
    "Price 40% above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pa40&ft=4",
    "Price 50% above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pa50&ft=4",
    "Price crossed SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pc&ft=4",
    "Price crossed SMA20 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pca&ft=4",
    "Price crossed SMA20 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_pcb&ft=4",
    "SMA20 crossed SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_cross50&ft=4",
    "SMA20 crossed SMA50 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_cross50a&ft=4",
    "SMA20 crossed SMA50 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_cross50b&ft=4",
    "SMA20 crossed SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_cross200&ft=4",
    "SMA20 crossed SMA200 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_cross200a&ft=4",
    "SMA20 crossed SMA200 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_cross200b&ft=4",
    "SMA20 above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_sa50&ft=4",
    "SMA20 below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_sb50&ft=4",
    "SMA20 above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_sa200&ft=4",
    "SMA20 below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma20_sb200&ft=4"
  },
  "20-Day High/Low": {
    "New High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_nh&ft=4",
    "New Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_nl&ft=4",
    "5% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b5h&ft=4",
    "10% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b10h&ft=4",
    "15% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b15h&ft=4",
    "20% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b20h&ft=4",
    "30% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b30h&ft=4",
    "40% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b40h&ft=4",
    "50% or more below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b50h&ft=4",
    "0-3% below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b0to3h&ft=4",
    "0-5% below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b0to5h&ft=4",
    "0-10% below High": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_b0to10h&ft=4",
    "5% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a5h&ft=4",
    "10% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a10h&ft=4",
    "15% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a15h&ft=4",
    "20% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a20h&ft=4",
    "30% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a30h&ft=4",
    "40% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a40h&ft=4",
    "50% or more above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a50h&ft=4",
    "0-3% above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a0to3h&ft=4",
    "0-5% above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a0to5h&ft=4",
    "0-10% above Low": "https://finviz.com/screener.ashx?v=111&f=ta_highlow20d_a0to10h&ft=4"
  },
  "Candlestick": {
    "Long Lower Shadow": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_lls&ft=4",
    "Long Upper Shadow": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_lus&ft=4",
    "Hammer": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_h&ft=4",
    "Inverted Hammer": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_ih&ft=4",
    "Spinning Top White": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_stw&ft=4",
    "Spinning Top Black": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_stb&ft=4",
    "Doji": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_d&ft=4",
    "Dragonfly Doji": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_dd&ft=4",
    "Gravestone Doji": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_gd&ft=4",
    "Marubozu White": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_mw&ft=4",
    "Marubozu Black": "https://finviz.com/screener.ashx?v=111&f=ta_candlestick_mb&ft=4"
  },
  "Forward P/E": {
    "Low (<15)": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_low&ft=4",
    "Profitable (>0)": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_profitable&ft=4",
    "High (>50)": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_high&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u5&ft=4",
    "Under 10": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u10&ft=4",
    "Under 15": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u15&ft=4",
    "Under 20": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u20&ft=4",
    "Under 25": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u25&ft=4",
    "Under 30": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u30&ft=4",
    "Under 35": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u35&ft=4",
    "Under 40": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u40&ft=4",
    "Under 45": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u45&ft=4",
    "Under 50": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_u50&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o5&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o10&ft=4",
    "Over 15": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o15&ft=4",
    "Over 20": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o20&ft=4",
    "Over 25": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o25&ft=4",
    "Over 30": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o30&ft=4",
    "Over 35": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o35&ft=4",
    "Over 40": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o40&ft=4",
    "Over 45": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o45&ft=4",
    "Over 50": "https://finviz.com/screener.ashx?v=111&f=fa_fpe_o50&ft=4"
  },
  "Price/Free Cash Flow": {
    "Low (<15)": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_low&ft=4",
    "High (>50)": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_high&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u5&ft=4",
    "Under 10": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u10&ft=4",
    "Under 15": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u15&ft=4",
    "Under 20": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u20&ft=4",
    "Under 25": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u25&ft=4",
    "Under 30": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u25&ft=4",
    "Under 35": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u25&ft=4",
    "Under 40": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u40&ft=4",
    "Under 45": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u45&ft=4",
    "Under 50": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u50&ft=4",
    "Under 60": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u60&ft=4",
    "Under 70": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u70&ft=4",
    "Under 80": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u80&ft=4",
    "Under 90": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u90&ft=4",
    "Under 100": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_u100&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o5&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o10&ft=4",
    "Over 15": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o15&ft=4",
    "Over 20": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o20&ft=4",
    "Over 25": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o25&ft=4",
    "Over 30": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o30&ft=4",
    "Over 35": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o35&ft=4",
    "Over 40": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o40&ft=4",
    "Over 45": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o45&ft=4",
    "Over 50": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o50&ft=4",
    "Over 60": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o60&ft=4",
    "Over 70": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o70&ft=4",
    "Over 80": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o80&ft=4",
    "Over 90": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o90&ft=4",
    "Over 100": "https://finviz.com/screener.ashx?v=111&f=fa_pfcf_o100&ft=4"
  },
  "Sales Growth (Past 5 Years)": {
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_neg&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_pos&ft=4",
    "Positive Low (0-10%)": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_poslow&ft=4",
    "High (>25%)": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_high&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_u5&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_u10&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_u15&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_u20&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_u25&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_u30&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_sales5years_o30&ft=4"
  },
  "Return on Assets": {
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_roa_pos&ft=4",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_roa_neg&ft=4",
    "Very Positive (>15%)": "https://finviz.com/screener.ashx?v=111&f=fa_roa_verypos&ft=4",
    "Very Negative (<=-15%)": "https://finviz.com/screener.ashx?v=111&f=fa_roa_veryneg&ft=4",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-50&ft=4",
    "Under -45%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-45&ft=4",
    "Under -40%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-40&ft=4",
    "Under -35%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-35&ft=4",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-30&ft=4",
    "Under -25%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-25&ft=4",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-20&ft=4",
    "Under -15%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-15&ft=4",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-10&ft=4",
    "Under -5%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_u-5&ft=4",
    "Over +5%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o5&ft=4",
    "Over +10%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o10&ft=4",
    "Over +15%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o15&ft=4",
    "Over +20%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o20&ft=4",
    "Over +25%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o25&ft=4",
    "Over +30%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o30&ft=4",
    "Over +35%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o35&ft=4",
    "Over +40%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o40&ft=4",
    "Over +45%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o45&ft=4",
    "Over +50%": "https://finviz.com/screener.ashx?v=111&f=fa_roa_o50&ft=4"
  },
  "LT Debt/Equity": {
    "High (>0.5)": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_high&ft=4",
    "Low (<0.1)": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_low&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u1&ft=4",
    "Under 0.9": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.9&ft=4",
    "Under 0.8": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.8&ft=4",
    "Under 0.7": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.7&ft=4",
    "Under 0.6": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.6&ft=4",
    "Under 0.5": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.5&ft=4",
    "Under 0.4": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.4&ft=4",
    "Under 0.3": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.3&ft=4",
    "Under 0.2": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.2&ft=4",
    "Under 0.1": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_u0.1&ft=4",
    "Over 0.1": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.1&ft=4",
    "Over 0.2": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.2&ft=4",
    "Over 0.3": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.3&ft=4",
    "Over 0.4": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.4&ft=4",
    "Over 0.5": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.5&ft=4",
    "Over 0.6": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.6&ft=4",
    "Over 0.7": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.7&ft=4",
    "Over 0.8": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.8&ft=4",
    "Over 0.9": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o0.9&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_ltdebteq_o1&ft=4"
  },
  "Payout Ratio": {
    "None (0%)": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_none&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_pos&ft=4",
    "Low (<20%)": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_low&ft=4",
    "High (>50%)": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_high&ft=4",
    "Over 0%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o0&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o10&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o20&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o30&ft=4",
    "Over 40%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o40&ft=4",
    "Over 50%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o50&ft=4",
    "Over 60%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o60&ft=4",
    "Over 70%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o70&ft=4",
    "Over 80%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o80&ft=4",
    "Over 90%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o90&ft=4",
    "Over 100%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_o100&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u10&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u20&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u30&ft=4",
    "Under 40%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u40&ft=4",
    "Under 50%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u50&ft=4",
    "Under 60%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u60&ft=4",
    "Under 70%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u70&ft=4",
    "Under 80%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u80&ft=4",
    "Under 90%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u90&ft=4",
    "Under 100%": "https://finviz.com/screener.ashx?v=111&f=fa_payoutratio_u100&ft=4"
  },
  "50-Day Simple Moving Average": {
    "Price below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pb&ft=4",
    "Price 10% below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pb10&ft=4",
    "Price 20% below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pb20&ft=4",
    "Price 30% below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pb30&ft=4",
    "Price 40% below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pb40&ft=4",
    "Price 50% below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pb50&ft=4",
    "Price above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa&ft=4",
    "Price 10% above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa10&ft=4",
    "Price 20% above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa20&ft=4",
    "Price 30% above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa30&ft=4",
    "Price 40% above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa40&ft=4",
    "Price 50% above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pa50&ft=4",
    "Price crossed SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pc&ft=4",
    "Price crossed SMA50 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pca&ft=4",
    "Price crossed SMA50 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_pcb&ft=4",
    "SMA50 crossed SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_cross20&ft=4",
    "SMA50 crossed SMA20 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_cross20a&ft=4",
    "SMA50 crossed SMA20 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_cross20b&ft=4",
    "SMA50 crossed SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_cross200&ft=4",
    "SMA50 crossed SMA200 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_cross200a&ft=4",
    "SMA50 crossed SMA200 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_cross200b&ft=4",
    "SMA50 above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_sa20&ft=4",
    "SMA50 below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_sb20&ft=4",
    "SMA50 above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_sa200&ft=4",
    "SMA50 below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma50_sb200&ft=4"
  },
  "Beta": {
    "Under 0": "https://finviz.com/screener.ashx?v=111&f=ta_beta_u0&ft=4",
    "Under 0.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_u0.5&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=ta_beta_u1&ft=4",
    "Under 1.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_u1.5&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=ta_beta_u2&ft=4",
    "Over 0": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o0&ft=4",
    "Over 0.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o0.5&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o1&ft=4",
    "Over 1.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o1.5&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o2&ft=4",
    "Over 2.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o2.5&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o3&ft=4",
    "Over 4": "https://finviz.com/screener.ashx?v=111&f=ta_beta_o4&ft=4",
    "0 to 0.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_0to0.5&ft=4",
    "0 to 1": "https://finviz.com/screener.ashx?v=111&f=ta_beta_0to1&ft=4",
    "0.5 to 1": "https://finviz.com/screener.ashx?v=111&f=ta_beta_0.5to1&ft=4",
    "0.5 to 1.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_0.5to1.5&ft=4",
    "1 to 1.5": "https://finviz.com/screener.ashx?v=111&f=ta_beta_1to1.5&ft=4",
    "1 to 2": "https://finviz.com/screener.ashx?v=111&f=ta_beta_1to2&ft=4"
  },
  "PEG": {
    "Low (<1)": "https://finviz.com/screener.ashx?v=111&f=fa_peg_low&ft=4",
    "High (>2)": "https://finviz.com/screener.ashx?v=111&f=fa_peg_high&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_peg_u1&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=fa_peg_u2&ft=4",
    "Under 3": "https://finviz.com/screener.ashx?v=111&f=fa_peg_u3&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_peg_o1&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=fa_peg_o2&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=fa_peg_o3&ft=4"
  },
  "EPS Growth This Year": {
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_neg&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_pos&ft=4",
    "Positive Low (0-10%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_poslow&ft=4",
    "High (>25%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_high&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_u5&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_u10&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_u15&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_u20&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_u25&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_u30&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy_o30&ft=4"
  },
  "Return on Equity": {
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_roe_pos&ft=4",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_roe_neg&ft=4",
    "Very Positive (>30%)": "https://finviz.com/screener.ashx?v=111&f=fa_roe_verypos&ft=4",
    "Very Negative (<-15%)": "https://finviz.com/screener.ashx?v=111&f=fa_roe_veryneg&ft=4",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-50&ft=4",
    "Under -45%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-45&ft=4",
    "Under -40%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-40&ft=4",
    "Under -35%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-35&ft=4",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-30&ft=4",
    "Under -25%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-25&ft=4",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-20&ft=4",
    "Under -15%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-15&ft=4",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-10&ft=4",
    "Under -5%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_u-5&ft=4",
    "Over +5%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o5&ft=4",
    "Over +10%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o10&ft=4",
    "Over +15%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o15&ft=4",
    "Over +20%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o20&ft=4",
    "Over +25%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o25&ft=4",
    "Over +30%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o30&ft=4",
    "Over +35%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o35&ft=4",
    "Over +40%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o40&ft=4",
    "Over +45%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o45&ft=4",
    "Over +50%": "https://finviz.com/screener.ashx?v=111&f=fa_roe_o50&ft=4"
  },
  "Debt/Equity": {
    "High (>0.5)": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_high&ft=4",
    "Low (<0.1)": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_low&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u1&ft=4",
    "Under 0.9": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.9&ft=4",
    "Under 0.8": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.8&ft=4",
    "Under 0.7": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.7&ft=4",
    "Under 0.6": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.6&ft=4",
    "Under 0.5": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.5&ft=4",
    "Under 0.4": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.4&ft=4",
    "Under 0.3": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.3&ft=4",
    "Under 0.2": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.2&ft=4",
    "Under 0.1": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_u0.1&ft=4",
    "Over 0.1": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.1&ft=4",
    "Over 0.2": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.2&ft=4",
    "Over 0.3": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.3&ft=4",
    "Over 0.4": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.4&ft=4",
    "Over 0.5": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.5&ft=4",
    "Over 0.6": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.6&ft=4",
    "Over 0.7": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.7&ft=4",
    "Over 0.8": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.8&ft=4",
    "Over 0.9": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o0.9&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_debteq_o1&ft=4"
  },
  "Insider Ownership": {
    "Low (<5%)": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_low&ft=4",
    "High (>30%)": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_high&ft=4",
    "Very High (>50%)": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_veryhigh&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o10&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o20&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o30&ft=4",
    "Over 40%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o40&ft=4",
    "Over 50%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o50&ft=4",
    "Over 60%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o60&ft=4",
    "Over 70%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o70&ft=4",
    "Over 80%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o80&ft=4",
    "Over 90%": "https://finviz.com/screener.ashx?v=111&f=sh_insiderown_o90&ft=4"
  },
  "Volatility (Week)": {
    "Over 3%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo3&ft=4",
    "Over 4%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo4&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo5&ft=4",
    "Over 6%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo6&ft=4",
    "Over 7%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo7&ft=4",
    "Over 8%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo8&ft=4",
    "Over 9%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo9&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo10&ft=4",
    "Over 12%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo12&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_wo15&ft=4"
  },
  "Volatility (Month)": {
    "Over 2%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo2&ft=4",
    "Over 3%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo3&ft=4",
    "Over 4%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo4&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo5&ft=4",
    "Over 6%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo6&ft=4",
    "Over 7%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo7&ft=4",
    "Over 8%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo8&ft=4",
    "Over 9%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo9&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo10&ft=4",
    "Over 12%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo12&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=ta_volatility_mo15&ft=4"
  },
  "200-Day Simple Moving Average": {
    "Price below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb&ft=4",
    "Price 10% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb10&ft=4",
    "Price 20% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb20&ft=4",
    "Price 30% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb30&ft=4",
    "Price 40% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb40&ft=4",
    "Price 50% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb50&ft=4",
    "Price 60% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb60&ft=4",
    "Price 70% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb70&ft=4",
    "Price 80% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb80&ft=4",
    "Price 90% below SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pb90&ft=4",
    "Price above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa&ft=4",
    "Price 10% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa10&ft=4",
    "Price 20% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa20&ft=4",
    "Price 30% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa30&ft=4",
    "Price 40% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa40&ft=4",
    "Price 50% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa50&ft=4",
    "Price 60% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa60&ft=4",
    "Price 70% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa70&ft=4",
    "Price 80% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa80&ft=4",
    "Price 90% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa90&ft=4",
    "Price 100% above SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pa90&ft=4",
    "Price crossed SMA200": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pc&ft=4",
    "Price crossed SMA200 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pca&ft=4",
    "Price crossed SMA200 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_pcb&ft=4",
    "SMA200 crossed SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_cross20&ft=4",
    "SMA200 crossed SMA20 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_cross20a&ft=4",
    "SMA200 crossed SMA20 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_cross20b&ft=4",
    "SMA200 crossed SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_cross50&ft=4",
    "SMA200 crossed SMA50 above": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_cross50a&ft=4",
    "SMA200 crossed SMA50 below": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_cross50b&ft=4",
    "SMA200 below SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_sb20&ft=4",
    "SMA200 above SMA20": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_sa20&ft=4",
    "SMA200 above SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_sa50&ft=4",
    "SMA200 below SMA50": "https://finviz.com/screener.ashx?v=111&f=ta_sma200_sb50&ft=4"
  },
  "Average True Range": {
    "Over 0.25": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o0.25&ft=4",
    "Over 0.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o0.5&ft=4",
    "Over 0.75": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o0.75&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o1&ft=4",
    "Over 1.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o1.5&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o2&ft=4",
    "Over 2.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o2.5&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o3&ft=4",
    "Over 3.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o3.5&ft=4",
    "Over 4": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o4&ft=4",
    "Over 4.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o4.5&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_o5&ft=4",
    "Under 0.25": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u0.25&ft=4",
    "Under 0.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u0.5&ft=4",
    "Under 0.75": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u0.75&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u1&ft=4",
    "Under 1.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u1.5&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u2&ft=4",
    "Under 2.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u2.5&ft=4",
    "Under 3": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u3&ft=4",
    "Under 3.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u3.5&ft=4",
    "Under 4": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u4&ft=4",
    "Under 4.5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u4.5&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=ta_averagetruerange_u5&ft=4"
  },
  "Price": {
    "Under $1": "https://finviz.com/screener.ashx?v=111&f=sh_price_u1&ft=4",
    "Under $2": "https://finviz.com/screener.ashx?v=111&f=sh_price_u2&ft=4",
    "Under $3": "https://finviz.com/screener.ashx?v=111&f=sh_price_u3&ft=4",
    "Under $4": "https://finviz.com/screener.ashx?v=111&f=sh_price_u4&ft=4",
    "Under $5": "https://finviz.com/screener.ashx?v=111&f=sh_price_u5&ft=4",
    "Under $7": "https://finviz.com/screener.ashx?v=111&f=sh_price_u7&ft=4",
    "Under $10": "https://finviz.com/screener.ashx?v=111&f=sh_price_u10&ft=4",
    "Under $15": "https://finviz.com/screener.ashx?v=111&f=sh_price_u15&ft=4",
    "Under $20": "https://finviz.com/screener.ashx?v=111&f=sh_price_u20&ft=4",
    "Under $30": "https://finviz.com/screener.ashx?v=111&f=sh_price_u30&ft=4",
    "Under $40": "https://finviz.com/screener.ashx?v=111&f=sh_price_u40&ft=4",
    "Under $50": "https://finviz.com/screener.ashx?v=111&f=sh_price_u50&ft=4",
    "Over $1": "https://finviz.com/screener.ashx?v=111&f=sh_price_o1&ft=4",
    "Over $2": "https://finviz.com/screener.ashx?v=111&f=sh_price_o2&ft=4",
    "Over $3": "https://finviz.com/screener.ashx?v=111&f=sh_price_o3&ft=4",
    "Over $4": "https://finviz.com/screener.ashx?v=111&f=sh_price_o4&ft=4",
    "Over $5": "https://finviz.com/screener.ashx?v=111&f=sh_price_o5&ft=4",
    "Over $7": "https://finviz.com/screener.ashx?v=111&f=sh_price_o7&ft=4",
    "Over $10": "https://finviz.com/screener.ashx?v=111&f=sh_price_o10&ft=4",
    "Over $15": "https://finviz.com/screener.ashx?v=111&f=sh_price_o15&ft=4",
    "Over $20": "https://finviz.com/screener.ashx?v=111&f=sh_price_o20&ft=4",
    "Over $30": "https://finviz.com/screener.ashx?v=111&f=sh_price_o30&ft=4",
    "Over $40": "https://finviz.com/screener.ashx?v=111&f=sh_price_o40&ft=4",
    "Over $50": "https://finviz.com/screener.ashx?v=111&f=sh_price_o50&ft=4",
    "Over $60": "https://finviz.com/screener.ashx?v=111&f=sh_price_o60&ft=4",
    "Over $70": "https://finviz.com/screener.ashx?v=111&f=sh_price_o70&ft=4",
    "Over $80": "https://finviz.com/screener.ashx?v=111&f=sh_price_o80&ft=4",
    "Over $90": "https://finviz.com/screener.ashx?v=111&f=sh_price_o90&ft=4",
    "Over $100": "https://finviz.com/screener.ashx?v=111&f=sh_price_o100&ft=4",
    "$1 to $5": "https://finviz.com/screener.ashx?v=111&f=sh_price_1to5&ft=4",
    "$5 to $10": "https://finviz.com/screener.ashx?v=111&f=sh_price_5to10&ft=4",
    "$5 to $20": "https://finviz.com/screener.ashx?v=111&f=sh_price_5to20&ft=4&o=vo",
    "$10 to $20": "https://finviz.com/screener.ashx?v=111&f=sh_price_10to20&ft=4",
    "$20 to $50": "https://finviz.com/screener.ashx?v=111&f=sh_price_20to50&ft=4",
    "$50 to $100": "https://finviz.com/screener.ashx?v=111&f=sh_price_50to100&ft=4"
  },
  "Price-to-Sales Ratio": {
    "Low (<1)": "https://finviz.com/screener.ashx?v=111&f=fa_ps_low&ft=4",
    "High (>10)": "https://finviz.com/screener.ashx?v=111&f=fa_ps_high&ft=4",
    "Under 1": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u1&ft=4",
    "Under 2": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u2&ft=4",
    "Under 3": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u3&ft=4",
    "Under 4": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u4&ft=4",
    "Under 5": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u5&ft=4",
    "Under 6": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u6&ft=4",
    "Under 7": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u7&ft=4",
    "Under 8": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u8&ft=4",
    "Under 9": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u9&ft=4",
    "Under 10": "https://finviz.com/screener.ashx?v=111&f=fa_ps_u10&ft=4",
    "Over 1": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o1&ft=4",
    "Over 2": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o2&ft=4",
    "Over 3": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o3&ft=4",
    "Over 4": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o4&ft=4",
    "Over 5": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o5&ft=4",
    "Over 6": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o6&ft=4",
    "Over 7": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o7&ft=4",
    "Over 8": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o8&ft=4",
    "Over 9": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o9&ft=4",
    "Over 10": "https://finviz.com/screener.ashx?v=111&f=fa_ps_o10&ft=4"
  },
  "EPS Growth Next Year": {
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_neg&ft=4",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_pos&ft=4",
    "Positive Low (0-10%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_poslow&ft=4",
    "High (>25%)": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_high&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_u5&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_u10&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_u15&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_u20&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_u25&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_u30&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_epsyoy1_o30&ft=4"
  },
  "Return on Investment": {
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_roi_pos&ft=4",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_roi_neg&ft=4",
    "Very Positive (>25%)": "https://finviz.com/screener.ashx?v=111&f=fa_roi_verypos&ft=4",
    "Very Negative (<-10%)": "https://finviz.com/screener.ashx?v=111&f=fa_roi_veryneg&ft=4",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-50&ft=4",
    "Under -45%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-45&ft=4",
    "Under -40%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-40&ft=4",
    "Under -35%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-35&ft=4",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-30&ft=4",
    "Under -25%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-25&ft=4",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-20&ft=4",
    "Under -15%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-15&ft=4",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-10&ft=4",
    "Under -5%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_u-5&ft=4",
    "Over +5%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o5&ft=4",
    "Over +10%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o10&ft=4",
    "Over +15%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o15&ft=4",
    "Over +20%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o20&ft=4",
    "Over +25%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o25&ft=4",
    "Over +30%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o30&ft=4",
    "Over +35%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o35&ft=4",
    "Over +40%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o40&ft=4",
    "Over +45%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o45&ft=4",
    "Over +50%": "https://finviz.com/screener.ashx?v=111&f=fa_roi_o50&ft=4"
  },
  "Gross Margin": {
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_pos&ft=4",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_neg&ft=4",
    "High (>50%)": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_high&ft=4",
    "Under 90%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u90&ft=4",
    "Under 80%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u80&ft=4",
    "Under 70%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u70&ft=4",
    "Under 60%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u60&ft=4",
    "Under 50%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u50&ft=4",
    "Under 45%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u45&ft=4",
    "Under 40%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u40&ft=4",
    "Under 35%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u35&ft=4",
    "Under 30%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u30&ft=4",
    "Under 25%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u25&ft=4",
    "Under 20%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u20&ft=4",
    "Under 15%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u15&ft=4",
    "Under 10%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u10&ft=4",
    "Under 5%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u5&ft=4",
    "Under 0%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u0&ft=4",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u-10&ft=4",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u-20&ft=4",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u-30&ft=4",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u-50&ft=4",
    "Under -70%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u-70&ft=4",
    "Under -100%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_u-100&ft=4",
    "Over 0%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o0&ft=4",
    "Over 5%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o5&ft=4",
    "Over 10%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o10&ft=4",
    "Over 15%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o15&ft=4",
    "Over 20%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o20&ft=4",
    "Over 25%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o25&ft=4",
    "Over 30%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o30&ft=4",
    "Over 35%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o35&ft=4",
    "Over 40%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o40&ft=4&o=volume",
    "Over 45%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o45&ft=4&o=volume",
    "Over 50%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o50&ft=4&o=volume",
    "Over 60%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o60&ft=4&o=volume",
    "Over 70%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o70&ft=4&o=volume",
    "Over 80%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o80&ft=4&o=volume",
    "Over 90%": "https://finviz.com/screener.ashx?v=111&f=fa_grossmargin_o90&ft=4&o=volume"
  },
  "Insider Transactions": {
    "Very Negative (<-20%)": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_veryneg&ft=4&o=volume",
    "Negative (<0%)": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_neg&ft=4&o=volume",
    "Positive (>0%)": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_pos&ft=4&o=volume",
    "Very Positive (>20%)": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_verypos&ft=4&o=volume",
    "Under -90%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-90&ft=4&o=volume",
    "Under -80%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-80&ft=4&o=volume",
    "Under -70%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-70&ft=4&o=volume",
    "Under -60%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-60&ft=4&o=volume",
    "Under -50%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-50&ft=4&o=volume",
    "Under -45%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-45&ft=4&o=volume",
    "Under -40%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-40&ft=4&o=volume",
    "Under -35%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-35&ft=4&o=volume",
    "Under -30%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-30&ft=4&o=volume",
    "Under -25%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-25&ft=4&o=volume",
    "Under -20%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-20&ft=4&o=volume",
    "Under -15%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-15&ft=4&o=volume",
    "Under -10%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-10&ft=4&o=volume",
    "Under -5%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_u-5&ft=4&o=volume",
    "Over +5%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o5&ft=4&o=volume",
    "Over +10%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o10&ft=4&o=volume",
    "Over +15%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o15&ft=4&o=volume",
    "Over +20%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o20&ft=4&o=volume",
    "Over +25%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o25&ft=4&o=volume",
    "Over +30%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o30&ft=4&o=volume",
    "Over +35%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o35&ft=4&o=volume",
    "Over +40%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o40&ft=4&o=volume",
    "Over +45%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o45&ft=4&o=volume",
    "Over +50%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o50&ft=4&o=volume",
    "Over +60%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o60&ft=4&o=volume",
    "Over +70%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o70&ft=4&o=volume",
    "Over +80%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o80&ft=4&o=volume",
    "Over +90%": "https://finviz.com/screener.ashx?v=111&f=sh_insidertrans_o90&ft=4&o=volume"
  },
  "Option/Short": {
    "Optionable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_option&ft=4&o=volume",
    "Shortable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_short&ft=4&o=volume",
    "Not optionable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_notoption&ft=4&o=volume",
    "Not shortable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_notshort&ft=4&o=volume",
    "Optionable and shortable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_optionshort&ft=4&o=volume",
    "Optionable and not shortable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_optionnotshort&ft=4&o=volume",
    "Not optionable and shortable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_notoptionshort&ft=4&o=volume",
    "Not optionable and not shortable": "https://finviz.com/screener.ashx?v=111&f=sh_opt_notoptionnotshort&ft=4&o=volume"
  },
  "RSI (14)": {
    "Overbought (90)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_ob90&ft=4&o=volume",
    "Overbought (80)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_ob80&ft=4&o=volume",
    "Overbought (70)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_ob70&ft=4&o=volume",
    "Overbought (60)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_ob60&ft=4&o=volume",
    "Oversold (40)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_os40&ft=4&o=volume",
    "Oversold (30)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_os30&ft=4&o=volume",
    "Oversold (20)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_os20&ft=4&o=volume",
    "Oversold (10)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_os10&ft=4&o=volume",
    "Not Overbought (<60)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_nob60&ft=4&o=volume",
    "Not Overbought (<50)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_nob50&ft=4&o=volume",
    "Not Oversold (>50)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_nos50&ft=4&o=volume",
    "Not Oversold (>40)": "https://finviz.com/screener.ashx?v=111&f=ta_rsi_nos40&ft=4&o=volume"
  }
}
"""

# --- Supabase (module-level) ---
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://snbkmsivcmsqrdtaotwy.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNuYmttc2l2Y21zcXJkdGFvdHd5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyMjMwNzYsImV4cCI6MjA3MDc5OTA3Nn0.kd3fFz01oJDkUdew3I3DKrWdhz_3EtqoQcjy2V56U6U")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None

def sign_up(email: str, password: str):
    if not supabase:
        return False, "Supabase not initialized"
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        # supabase.auth.sign_up may return an object or dict depending on version
        if isinstance(res, dict):
            user = res.get('user') or res
        else:
            user = getattr(res, 'user', None) or res
        try:
            log_debug(f"sign_up response: {str(res)}")
            if user and user.get('id'):
                supabase.table('users').upsert({
                    'id': user['id'],
                    'email': email,
                    'credits': 300
                }).execute()
        except Exception:
            log_debug("sign_up: upsert/create user record failed", exc_info=True)
        return True, user
    except Exception as e:
        log_debug(f"sign_up failed for {email}: {e}", exc_info=True)
        return False, str(e)

def sign_in(email: str, password: str):
    if not supabase:
        return False, "Supabase not initialized"
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if isinstance(res, dict):
            user = res.get('user') or res
        else:
            user = getattr(res, 'user', None) or res
        log_debug(f"sign_in response for {email}: {str(res)}")
        return True, user
    except Exception as e:
        log_debug(f"sign_in failed for {email}: {e}", exc_info=True)
        return False, str(e)

def sign_out():
    if not supabase:
        return False
    try:
        supabase.auth.sign_out()
        return True
    except Exception:
        return False

def get_user_record(user_id: str):
    if not supabase:
        return None
    try:
        res = supabase.table('users').select('*').eq('id', user_id).single().execute()
        data = res.get('data') if isinstance(res, dict) else getattr(res, 'data', None)
        return data
    except Exception:
        return None

def create_user_record_if_missing(user_id: str, email: str):
    if not supabase:
        return
    try:
        existing = get_user_record(user_id)
        if not existing:
            supabase.table('users').insert({'id': user_id, 'email': email, 'credits': 300}).execute()
    except Exception:
        pass

def get_credits(user_id: str):
    rec = get_user_record(user_id)
    if rec and 'credits' in rec:
        return int(rec['credits'])
    return None

def deduct_credits(user_id: str, amount: int):
    if not supabase:
        return False, "Supabase not initialized"
    try:
        rec = get_user_record(user_id)
        if not rec:
            return False, "User record not found"
        current = int(rec.get('credits', 0))
        if current < amount:
            return False, "Insufficient credits"
        new = current - amount
        supabase.table('users').update({'credits': new}).eq('id', user_id).execute()
        return True, new
    except Exception as e:
        return False, str(e)


# -------- Local session persistence helpers (simple, local-file based) --------
def save_local_session(user_obj):
    """Save a minimal user session to a local file so it persists across page reloads (local dev only)."""
    try:
        if not user_obj:
            return
        # Normalize to a simple dict with id and email and any token if present
        if isinstance(user_obj, dict):
            data = {
                'id': user_obj.get('id'),
                'email': user_obj.get('email'),
            }
            # optionally keep access_token if present
            if user_obj.get('access_token'):
                data['access_token'] = user_obj.get('access_token')
        else:
            data = {
                'id': getattr(user_obj, 'id', None),
                'email': getattr(user_obj, 'email', None),
            }
            token = getattr(user_obj, 'access_token', None)
            if token:
                data['access_token'] = token

        with open('.session.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass

def load_local_session():
    """Load the saved local session if present."""
    try:
        if os.path.exists('.session.json'):
            with open('.session.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None

def delete_local_session():
    try:
        if os.path.exists('.session.json'):
            os.remove('.session.json')
    except Exception:
        pass

# Load persisted session on startup (if any)
try:
    if 'user' not in st.session_state or not st.session_state.get('user'):
        loaded = load_local_session()
        if loaded:
            st.session_state['user'] = loaded
except Exception:
    pass

# -------- Supabase refresh-token storage (less secure, server-side DB) --------
def save_refresh_token_to_db(user_id: str, refresh_token: str):
    """Store the refresh token in Supabase in a safe table (remember_tokens) or users table as fallback."""
    if not supabase or not user_id or not refresh_token:
        return False
    try:
        # Try upsert into remember_tokens table
        try:
            supabase.table('remember_tokens').upsert({'id': user_id, 'refresh_token': refresh_token}).execute()
            return True
        except Exception:
            # Fallback to update users table if present
            try:
                supabase.table('users').update({'refresh_token': refresh_token}).eq('id', user_id).execute()
                return True
            except Exception:
                return False
    except Exception:
        return False

def get_refresh_token_from_db(user_id: str):
    if not supabase or not user_id:
        return None
    try:
        try:
            res = supabase.table('remember_tokens').select('refresh_token').eq('id', user_id).single().execute()
            data = res.get('data') if isinstance(res, dict) else getattr(res, 'data', None)
            if data and 'refresh_token' in data:
                return data['refresh_token']
        except Exception:
            # fallback to users table
            try:
                res = supabase.table('users').select('refresh_token').eq('id', user_id).single().execute()
                data = res.get('data') if isinstance(res, dict) else getattr(res, 'data', None)
                if data and 'refresh_token' in data:
                    return data['refresh_token']
            except Exception:
                return None
    except Exception:
        return None
    return None

def delete_refresh_token_from_db(user_id: str):
    if not supabase or not user_id:
        return False
    try:
        try:
            supabase.table('remember_tokens').delete().eq('id', user_id).execute()
        except Exception:
            try:
                supabase.table('users').update({'refresh_token': None}).eq('id', user_id).execute()
            except Exception:
                pass
        return True
    except Exception:
        return False

# Helper: refresh session via Supabase REST token endpoint using a refresh token
def refresh_session_with_refresh_token(refresh_token: str):
    try:
        if not refresh_token:
            return None
        token_url = SUPABASE_URL.rstrip('/') + '/auth/v1/token'
        headers = {
            'apikey': SUPABASE_KEY,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = f'grant_type=refresh_token&refresh_token={refresh_token}'
        try:
            log_debug("refresh_session_with_refresh_token: calling token endpoint (token redacted)")
            resp = requests.post(token_url, headers=headers, data=data, timeout=10)
            log_debug(f"refresh token response status: {resp.status_code} content: {resp.text}")
            if resp.status_code == 200:
                return resp.json()
            else:
                return None
        except Exception as e:
            log_debug(f"refresh_session_with_refresh_token request failed: {e}", exc_info=True)
            return None
    except Exception:
        log_debug("refresh_session_with_refresh_token: unexpected failure", exc_info=True)
        return None

# If the browser redirected with a remember_uid query param, try to restore session
try:
    params = st.query_params
    remember_uid = params.get('remember_uid', [None])[0] if params else None
    if remember_uid and (not st.session_state.get('user')):
        # Attempt to load refresh token from DB and refresh session
        rt = get_refresh_token_from_db(remember_uid)
        if rt:
            refreshed = refresh_session_with_refresh_token(rt)
            if refreshed and isinstance(refreshed, dict):
                # extract user object and set session
                user = refreshed.get('user') or refreshed.get('user')
                if user:
                    st.session_state['user'] = user
    # Handle logout via query param ?action=logout
    action = params.get('action', [None])[0] if params else None
    if action == 'logout':
        try:
            sign_out()
        except Exception:
            pass
        try:
            st.session_state.pop('user', None)
        except Exception:
            pass
        try:
            delete_local_session()
        except Exception:
            pass
except Exception:
    pass


def generate_ai_summary(content_data, ticker, sentiment_score, stats=None, top_pos=None, top_neg=None):
    """
    Genera un resumen inteligente usando OpenRouter AI
    """
    try:
        # Configure OpenAI client (use OPENAI_API_KEY from environment)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "AI service not configured. Set OPENAI_API_KEY in your .env or environment."
        base_url = os.getenv('OPENAI_API_BASE')
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        # Preparar el contexto para el AI
        full_content = "\n".join([f"- {item['title']}: {item['content'][:700]}..." for item in content_data[:8]])

        stats_section = ""
        if stats:
            stats_section = (
                f"Articles analyzed: {stats.get('articles_count', 'N/A')}\n"
                f"Full content articles: {stats.get('full_count', 'N/A')}\n"
                f"Content completeness rate: {stats.get('success_rate', 'N/A'):.1f}%\n"
                f"Press releases detected: {stats.get('press_releases', 0)}\n"
            )

        # Prepare top headlines lists
        top_pos_section = "" if not top_pos else "\nTop positive headlines:\n" + "\n".join([f"- {h}" for h in top_pos[:3]])
        top_neg_section = "" if not top_neg else "\nTop negative headlines:\n" + "\n".join([f"- {h}" for h in top_neg[:3]])

        prompt = f"""
        You are an expert financial analyst. Analyze the following news about {ticker} and produce a structured, in-depth analysis.

        CONTEXT STATS:
        {stats_section}

        TOP HEADLINES:
        {top_pos_section}
        {top_neg_section}

        RAW CONTENT (first {min(8, len(content_data))} items):
        {full_content}

        CURRENT AGGREGATE SENTIMENT: {sentiment_score:.3f} (-1 very negative, +1 very positive)

        Please provide a detailed answer that includes:
        1) Key statistics and a short interpretation (highlight what matters most)\n
        2) 3-5 main topics or drivers extracted from the headlines and content (bullet list)\n
        3) Potential impact on the stock price and which scenarios would lead to up/down movements\n
        4) List of the most relevant articles (title + 1-sentence why it's important)\n
        5) Actionable monitoring checklist (3 concrete signals to watch)\n
        Prefer a structured response with short paragraphs and clear headings. If the analysis is long, expand the response and include a short 2-3 sentence executive summary at the top.

        Return the answer in plain text but format headings with ALL CAPS (e.g., SUMMARY:, TOPICS:, IMPACT:).

        Keep the language English and be concise but thorough (aim for ~350-700 words if needed).
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst. Provide clear, structured analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.2
        )

        text = response.choices[0].message.content
        # Wrap in HTML with a slightly larger container to allow expansion
        html = f"""
        <div style='background: #071018; padding:18px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); max-width:100%; box-sizing:border-box;'>
            <div style='color:#fff; font-weight:700; margin-bottom:8px;'>AI ANALYSIS FOR {ticker.upper()}</div>
            <div style='color:#cbd5e1; white-space:pre-wrap; line-height:1.45;'>{text}</div>
        </div>
        """
        return html
        
    except Exception as e:
        # log full traceback and return a slightly more informative message
        log_debug(f"generate_ai_summary failed for {ticker}: {e}", exc_info=True)
        return f"Error generating AI analysis: {str(e)} (see {LOG_FILE} for details)"

def load_finviz_filters():
    """
    Load Finviz filters from JSON file
    """
    try:
        # If the developer pasted the full filtros.json into this file, parse it first
        if EMBEDDED_FINVIZ_FILTERS_JSON and EMBEDDED_FINVIZ_FILTERS_JSON.strip() and 'PASTE YOUR filtros.json' not in EMBEDDED_FINVIZ_FILTERS_JSON:
            try:
                return json.loads(EMBEDDED_FINVIZ_FILTERS_JSON)
            except Exception as e:
                log_debug(f"Embedded filtros.json parse failed: {e}", exc_info=True)
                # fall through to disk-based loading
        # Resolve filtros.json relative to this file to avoid CWD-related FileNotFoundError
        base = Path(__file__).resolve().parent
        filtros_path = base / 'filtros.json'
        if not filtros_path.exists():
            # Try also repository root as fallback
            filtros_path = Path('filtros.json')
        with open(filtros_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_debug(f"Error loading filtros.json at {filtros_path if 'filtros_path' in locals() else 'unknown'}: {e}", exc_info=True)
        # Show a user-friendly message but avoid exposing stack trace
        st.error(f"Error loading filters: {str(e)}")
        return {}

def generate_investment_advice(user_message, investment_profile=None, chat_history=None):
    """
    Generate investment advice using GPT-4.1-nano with Finviz integration and conversation context
    """
    try:
        # Configure OpenAI client (use OPENAI_API_KEY from environment)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "AI service not configured. Set OPENAI_API_KEY in your .env or environment."
        base_url = os.getenv('OPENAI_API_BASE')
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        
        # Load Finviz filters
        filters = load_finviz_filters()
        
        # Create context about available filters and strategies
        filters_context = f"""
        Available Finviz filters: {json.dumps(filters, indent=2)}
        
        Investment Profile Context: {investment_profile if investment_profile else 'Not specified'}
        """
        
        # Build conversation context from chat history
        conversation_context = ""
        if chat_history and len(chat_history) > 0:
            conversation_context = "\n\nPrevious conversation context:\n"
            for msg in chat_history[-6:]:  # Last 6 messages for context (3 exchanges)
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\nCurrent question:\n"
        
        # Create comprehensive prompt with conversation context
        system_prompt = f"""
        You are an expert investment advisor specializing in Finviz stock screening and put options strategies.
        
        {filters_context}
        
        Guidelines:
        1. Maintain conversation continuity - reference previous discussions when relevant
        2. Remember user preferences and investment style mentioned earlier
        3. Build upon previous recommendations and refine them based on new questions
        4. Use Finviz filters to create specific stock screening strategies
        5. Generate direct Finviz URLs when appropriate
        6. Focus on cash-secured puts and covered put strategies
        7. Explain each filter and its relevance
        8. Provide step-by-step guidance
        9. Keep responses concise and actionable
        10. If user asks follow-up questions, reference your previous advice and expand on it
        
        {conversation_context}
        """
        
        # Prepare messages for the API
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history to maintain context
        if chat_history:
            for msg in chat_history[-8:]:  # Last 8 messages for better context
                messages.append({
                    "role": msg["role"], 
                    "content": msg["content"]
                })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        log_debug(f"generate_investment_advice failed: {e}", exc_info=True)
        return f"Error generating investment advice: {str(e)} (see {LOG_FILE} for details)"

def show_investment_chatbot():
    """
    Display investment strategy chatbot interface - Modern clean design
    """
    # --- Supabase configuration (Auth + credits) ---
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://snbkmsivcmsqrdtaotwy.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNuYmttc2l2Y21zcXJkdGFvdHd5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyMjMwNzYsImV4cCI6MjA3MDc5OTA3Nn0.kd3fFz01oJDkUdew3I3DKrWdhz_3EtqoQcjy2V56U6U")

    # Initialize supabase client
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Supabase client init failed: {e}")
        supabase = None

    # Helper functions for auth and credits
    def sign_up(email: str, password: str):
        if not supabase:
            return False, "Supabase not initialized"
        try:
            res = supabase.auth.sign_up({"email": email, "password": password})
            user = res.get('user') or res
            # create users record with 300 credits (if table exists)
            try:
                if user and user.get('id'):
                    supabase.table('users').upsert({
                        'id': user['id'],
                        'email': email,
                        'credits': 300
                    }).execute()
            except Exception:
                pass
            return True, user
        except Exception as e:
            return False, str(e)

    def sign_in(email: str, password: str):
        if not supabase:
            return False, "Supabase not initialized"
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            user = res.get('user') or res
            return True, user
        except Exception as e:
            return False, str(e)

    def sign_out():
        if not supabase:
            return False
        try:
            supabase.auth.sign_out()
            return True
        except Exception:
            return False

    def get_user_record(user_id: str):
        if not supabase:
            return None
        try:
            res = supabase.table('users').select('*').eq('id', user_id).single().execute()
            data = res.get('data') if isinstance(res, dict) else getattr(res, 'data', None)
            return data
        except Exception:
            return None

    def create_user_record_if_missing(user_id: str, email: str):
        if not supabase:
            return
        try:
            existing = get_user_record(user_id)
            if not existing:
                supabase.table('users').insert({'id': user_id, 'email': email, 'credits': 300}).execute()
        except Exception:
            pass

    def get_credits(user_id: str):
        rec = get_user_record(user_id)
        if rec and 'credits' in rec:
            return int(rec['credits'])
        return None

    def deduct_credits(user_id: str, amount: int):
        if not supabase:
            return False, "Supabase not initialized"
        try:
            # Atomically decrement credits using RPC would be ideal; fallback to read-update
            rec = get_user_record(user_id)
            if not rec:
                return False, "User record not found"
            current = int(rec.get('credits', 0))
            if current < amount:
                return False, "Insufficient credits"
            new = current - amount
            supabase.table('users').update({'credits': new}).eq('id', user_id).execute()
            return True, new
        except Exception as e:
            return False, str(e)

    # Back button and title (responsive + centered)
    st.markdown(
        """
        <style>
        .page-title { 
            color: white; 
            margin: 0; 
            font-size: clamp(1.25rem, 2.5vw, 2rem);
            text-align: center;
            line-height: 1.2;
            word-break: break-word;
        }
        
        .welcome-section {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, rgba(16, 163, 127, 0.05), rgba(0, 0, 0, 0.3));
            border-radius: 15px;
            margin: 2rem 0;
            border: 1px solid #333;
        }
        
        .chat-container {
            background: linear-gradient(135deg, rgba(16, 163, 127, 0.1), rgba(0, 0, 0, 0.8));
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            border: 1px solid #333;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .user-message {
            background: linear-gradient(135deg, #2d2d2d, #1a1a1a);
            padding: 20px 25px;
            border-radius: 20px;
            margin: 15px 0;
            margin-left: 80px;
            color: white;
            border: 1px solid #404040;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            position: relative;
        }
        
        .user-message::before {
            content: "👤";
            position: absolute;
            left: -25px;
            top: 15px;
            font-size: 1.2rem;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #1a1a1a, #0d1117);
            padding: 20px 25px;
            border-radius: 20px;
            margin: 15px 0;
            margin-right: 80px;
            color: #e6e6e6;
            border: 1px solid #333;
            border-left: 4px solid #10a37f;
            box-shadow: 0 4px 15px rgba(16, 163, 127, 0.1);
            position: relative;
        }
        
        .assistant-message::before {
            content: "🤖";
            position: absolute;
            right: -25px;
            top: 15px;
            font-size: 1.2rem;
        }
        
        .input-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 2rem 0;
            border: 1px solid #404040;
            backdrop-filter: blur(10px);
            position: sticky;
            bottom: 0;
            z-index: 10;
        }
        
        /* Ocultar sección de configuración de perfil */
        .stExpander, [data-testid="expander"], details {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Ocultar expander y elementos relacionados */
        details[data-testid="expander"] {
            display: none !important;
        }
        
        /* Ocultar cualquier elemento que contenga "Configure Investment Profile" */
        *:has-text("Configure Investment Profile") {
            display: none !important;
        }
        
        /* Ocultar selectbox de Investment Style y Risk Tolerance */
        .stSelectbox {
            display: none !important;
        }
        
        /* Ocultar todo el expandir/contraer */
        summary {
            display: none !important;
        }
        
        /* Ocultar elementos con texto específico usando nth-child */
        div:contains("⚙️ Configure Investment Profile") {
            display: none !important;
        }
        
        div:contains("Configure your investment preferences") {
            display: none !important;
        }
        
        div:contains("Investment Style:") {
            display: none !important;
        }
        
        div:contains("Risk Tolerance:") {
            display: none !important;
        }
        
        /* Ocultar TODOS los elementos relacionados con configuración */
        *[data-testid*="expander"], 
        *[class*="expander"],
        *[class*="streamlit-expanderHeader"],
        *[class*="streamlit-expanderContent"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            overflow: hidden !important;
        }
        
        /* Ocultar por contenido usando pseudo-selectores más agresivos */
        p:contains("Configure your investment preferences") {
            display: none !important;
        }
        
        /* Forzar ocultación de cualquier texto relacionado */
        *:contains("⚙️") {
            display: none !important;
        }
        
        *:contains("Configure Investment") {
            display: none !important;
        }
        
        *:contains("personalized recommendations") {
            display: none !important;
        }
        
        /* Mejorar input area estilo ChatGPT */
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid #404040 !important;
            border-radius: 12px !important;
            color: white !important;
            resize: none !important;
            font-size: 16px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.2s ease !important;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #10a37f !important;
            box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
            outline: none !important;
        }
        
        .stTextArea > div > div > textarea::placeholder {
            color: #888 !important;
            font-style: italic !important;
        }
        
        /* Botones estilo ChatGPT */
        .stButton > button {
            background: #10a37f !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            min-height: 44px !important;
        }
        
        .stButton > button:hover {
            background: #0d8a66 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(16, 163, 127, 0.3) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Botón Clear con estilo diferente */
        .stButton:last-child > button {
            background: rgba(255, 255, 255, 0.1) !important;
            color: #ccc !important;
            border: 1px solid #404040 !important;
        }
        
        .stButton:last-child > button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border-color: #666 !important;
        }
        
        /* Input container más limpio */
        .input-container {
            display: flex;
            align-items: flex-end;
            gap: 12px;
            max-width: 100%;
            background: rgba(255, 255, 255, 0.02);
            padding: 16px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .button-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 70px;
        }
        
        /* Mejorar responsividad */
        @media (max-width: 768px) {
            .input-container {
                flex-direction: column;
                gap: 10px;
            }
            
            .button-group {
                flex-direction: row;
                justify-content: center;
                min-width: auto;
                width: 100%;
            }
            
            .stButton > button {
                min-height: 40px !important;
                font-size: 14px !important;
            }
        }
        
        /* Animación suave para el input */
        .stTextArea > div > div {
            transition: all 0.2s ease !important;
        }
        
        /* Placeholder mejorado */
        .stTextArea > div > div > textarea::placeholder {
            color: #666 !important;
            font-weight: 400 !important;
        }
        </style>
        
        <script>
        // JavaScript para ocultar elementos específicos después de la carga
        setTimeout(function() {
            // Ocultar todos los expanders
            const expanders = document.querySelectorAll('[data-testid="expander"]');
            expanders.forEach(el => el.style.display = 'none');
            
            // Ocultar elementos por texto específico - MÁS AGRESIVO
            const elements = document.querySelectorAll('*');
            elements.forEach(el => {
                if (el.innerText && (
                    el.innerText.includes('⚙️') ||
                    el.innerText.includes('Configure Investment') ||
                    el.innerText.includes('Configure your investment') ||
                    el.innerText.includes('personalized recommendations') ||
                    el.innerText.includes('Investment Style') ||
                    el.innerText.includes('Risk Tolerance')
                )) {
                    el.style.display = 'none !important';
                    el.style.visibility = 'hidden !important';
                    el.style.height = '0px !important';
                    el.style.overflow = 'hidden !important';
                    // Ocultar también elementos padre
                    if (el.parentNode) {
                        el.parentNode.style.display = 'none !important';
                        if (el.parentNode.parentNode) {
                            el.parentNode.parentNode.style.display = 'none !important';
                        }
                    }
                }
            });
            
            // Ocultar selectboxes
            const selectboxes = document.querySelectorAll('.stSelectbox');
            selectboxes.forEach(el => el.style.display = 'none');
            
        }, 500);
        
        // Ejecutar también cada vez que haya cambios en la página - MÁS AGRESIVO
        const observer = new MutationObserver(function() {
            // Ocultar expanders
            const expanders = document.querySelectorAll('[data-testid="expander"], details, summary');
            expanders.forEach(el => {
                el.style.display = 'none !important';
                el.style.visibility = 'hidden !important';
            });
            
            // Ocultar selectboxes
            const selectboxes = document.querySelectorAll('.stSelectbox');
            selectboxes.forEach(el => {
                el.style.display = 'none !important';
                el.style.visibility = 'hidden !important';
            });
            
            // Búsqueda agresiva de texto
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {
                if (el.innerText && (
                    el.innerText.includes('⚙️') ||
                    el.innerText.includes('Configure Investment') ||
                    el.innerText.includes('personalized recommendations')
                )) {
                    el.style.display = 'none !important';
                    el.style.visibility = 'hidden !important';
                }
            });
        });
        
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_center, col_right = st.columns([1, 8, 1])
    with col_left:
        if st.button("← Back", type="secondary"):
            st.session_state.current_page = "landing"
            try:
                st.experimental_set_query_params(page="landing")
            except Exception:
                pass
            st.experimental_rerun()
    with col_center:
        st.markdown('<h1 class="page-title">🎯 Investment Strategy Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Info panel: explain credits, features and limits (English)
    # compute current user's credits if available so we can show a badge on the investment page
    inv_uid = None
    inv_credits = None
    if 'user' in st.session_state and st.session_state.get('user'):
        u = st.session_state.get('user')
        inv_uid = u.get('id') if isinstance(u, dict) else getattr(u, 'id', None)
        try:
            inv_credits = get_credits(inv_uid) if inv_uid else None
        except Exception:
            inv_credits = None

    # --- Finviz snapshot integration for this page ---
    try:
        # local optional imports; scripts package added earlier
        from scripts.scrape_finviz import scrape_finviz_snapshot
        from scripts.finviz_chat import chat_with_snapshot
    except Exception:
        scrape_finviz_snapshot = None
        chat_with_snapshot = None

    # The investment page no longer shows a ticker input or fetch snapshot button.
    # Snapshot functionality has been moved to the KPI Dashboard page.

    info_html = """
    <div id="investment-info" style="background: rgba(2,6,23,0.8); color: #d1d5db; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <h3 style="color:#10a37f; margin-top:0; margin-bottom:0;">About the Investment Strategy Assistant</h3>
            <div id='credits-bubble-investment' style='background:linear-gradient(90deg,#0ea37f,#076b57);padding:6px 10px;border-radius:999px;color:#fff;font-weight:700;box-shadow:0 4px 12px rgba(6,45,40,0.25);font-size:0.95rem;'>Credits: {credits}</div>
        </div>
        <ul style="line-height:1.5; color: #cbd5e1; margin-top:0.75rem;">
            <li><strong>Cost:</strong> Each AI query costs <strong>100 credits</strong>.</li>
            <li><strong>Free credits:</strong> New users receive <strong>300 credits</strong> after confirming their registration email.</li>
            <li><strong>Conversation limit:</strong> Sessions are limited to <strong>5 exchanges</strong> to keep responses focused and fast.</li>
            <li><strong>Credits handling:</strong> Credits are deducted when a request is sent; if the analysis fails we attempt to refund automatically.</li>
            <li><strong>Features:</strong> Finviz screening, put/option strategy ideas, dividend & growth screens, personalized recommendations based on your profile.</li>
            <li><strong>Privacy & content:</strong> We do not display internal scraper details to users; only user-facing, client-ready analysis is shown.</li>
        </ul>
        <p style="color:#9aa; margin:0.5rem 0 0;">Questions? Ask here in chat or contact <a href="mailto:stockfeels@gmail.com">stockfeels@gmail.com</a>.</p>
    </div>
    """

    st.markdown(info_html.format(credits=(inv_credits if inv_credits is not None else 'N/A')), unsafe_allow_html=True)

    # Initialize chat history
    if "investment_chat_history" not in st.session_state:
        st.session_state.investment_chat_history = []
    
    # Show conversation context indicator
    if len(st.session_state.investment_chat_history) > 0:
        st.markdown(f"""
        <div style="background: rgba(16, 163, 127, 0.1); padding: 0.5rem 1rem; border-radius: 8px; margin: 1rem 0; border-left: 3px solid #10a37f;">
            <small style="color: #10a37f;">
                🧠 <strong>Context Aware:</strong> This conversation has {len(st.session_state.investment_chat_history)//2} exchanges. 
                The AI remembers your preferences and previous discussions.
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    # Show welcome section or chat history
    if len(st.session_state.investment_chat_history) == 0:
        # Welcome section with examples
        st.markdown("""
        <div class="welcome-section">
            <h2 style="color: white; margin-bottom: 1rem; font-size: 2rem;">Welcome to Investment Strategy Assistant</h2>
            <p style="color: #ccc; font-size: 1.1rem; margin-bottom: 2rem;">
                Get personalized Finviz screening strategies and put options guidance. 
                AI-powered recommendations tailored to your investment style and risk tolerance.
            </p>
            <p style="color: #10a37f; font-weight: 600;">Choose an example below or ask your own question:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💎 Value Stocks for Puts", use_container_width=True, type="secondary"):
                # enforce exchanges limit
                exchanges = len(st.session_state.investment_chat_history) // 2
                if exchanges >= 5:
                    st.error('Conversation limit reached (max 5 exchanges). Start a new chat to continue.')
                else:
                    quick_message = "Help me find undervalued large-cap stocks suitable for cash-secured puts with good fundamentals"
                    # Authentication & credits check
                    if 'user' not in st.session_state or not st.session_state.get('user'):
                        st.error('You must be logged in to use this feature.')
                    else:
                        user = st.session_state.get('user')
                        uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                        credits = get_credits(uid) if uid else None
                        COST = 100
                        if credits is None or credits < COST:
                            st.error(f'Insufficient credits. {COST} required to use this feature.')
                        else:
                            # deduct credits
                            ok, new_or_msg = deduct_credits(uid, COST)
                            if not ok:
                                st.error(f'Could not deduct credits: {new_or_msg}')
                            else:
                                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                                try:
                                    with st.spinner("🤔 Analyzing..."):
                                        ai_response = generate_investment_advice(
                                            quick_message,
                                            "Value Investing, Conservative",
                                            st.session_state.investment_chat_history[:-1]
                                        )
                                    st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                                    st.experimental_rerun()
                                except Exception as e:
                                    # rollback credits on failure
                                    try:
                                        supabase.table('users').update({'credits': new_or_msg}).eq('id', uid).execute()
                                    except Exception:
                                        pass
                                    st.error('Error during analysis. Credits refunded.')
        
        with col2:
            if st.button("💰 High Dividend Strategy", use_container_width=True, type="secondary"):
                exchanges = len(st.session_state.investment_chat_history) // 2
                if exchanges >= 5:
                    st.error('Conversation limit reached (max 5 exchanges). Start a new chat to continue.')
                else:
                    quick_message = "Show me high-dividend stocks perfect for generating income with put strategies"
                    if 'user' not in st.session_state or not st.session_state.get('user'):
                        st.error('You must be logged in to use this feature.')
                    else:
                        user = st.session_state.get('user')
                        uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                        credits = get_credits(uid) if uid else None
                        COST = 100
                        if credits is None or credits < COST:
                            st.error(f'Insufficient credits. {COST} required to use this feature.')
                        else:
                            ok, new_or_msg = deduct_credits(uid, COST)
                            if not ok:
                                st.error(f'Could not deduct credits: {new_or_msg}')
                            else:
                                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                                try:
                                    with st.spinner("🤔 Analyzing..."):
                                        ai_response = generate_investment_advice(
                                            quick_message,
                                            "Dividend Income, Conservative",
                                            st.session_state.investment_chat_history[:-1]
                                        )
                                    st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                                    st.experimental_rerun()
                                except Exception as e:
                                    try:
                                        supabase.table('users').update({'credits': new_or_msg}).eq('id', uid).execute()
                                    except Exception:
                                        pass
                                    st.error('Error during analysis. Credits refunded.')
        
        with col3:
            if st.button("⚡ Growth Screening", use_container_width=True, type="secondary"):
                exchanges = len(st.session_state.investment_chat_history) // 2
                if exchanges >= 5:
                    st.error('Conversation limit reached (max 5 exchanges). Start a new chat to continue.')
                else:
                    quick_message = "Create a Finviz filter for growth stocks suitable for covered put strategies"
                    if 'user' not in st.session_state or not st.session_state.get('user'):
                        st.error('You must be logged in to use this feature.')
                    else:
                        user = st.session_state.get('user')
                        uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                        credits = get_credits(uid) if uid else None
                        COST = 100
                        if credits is None or credits < COST:
                            st.error(f'Insufficient credits. {COST} required to use this feature.')
                        else:
                            ok, new_or_msg = deduct_credits(uid, COST)
                            if not ok:
                                st.error(f'Could not deduct credits: {new_or_msg}')
                            else:
                                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                                try:
                                    with st.spinner("🤔 Analyzing..."):
                                        ai_response = generate_investment_advice(
                                            quick_message,
                                            "Growth Investing, Moderate",
                                            st.session_state.investment_chat_history[:-1]
                                        )
                                    st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                                    st.experimental_rerun()
                                except Exception as e:
                                    try:
                                        supabase.table('users').update({'credits': new_or_msg}).eq('id', uid).execute()
                                    except Exception:
                                        pass
                                    st.error('Error during analysis. Credits refunded.')
    
    else:
        # Chat messages container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.investment_chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong style="color: #10a37f;">You</strong><br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong style="color: #10a37f;">AI Assistant</strong><br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Input container
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        user_input = st.text_area(
            "",
            placeholder="💬 Type your investment question here...",
            height=60,
            label_visibility="collapsed",
            key="user_input",
            help="Ask about Finviz screening, put options, or investment strategies"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="button-group">', unsafe_allow_html=True)
        send_button = st.button("Send", use_container_width=True, type="primary")
        clear_button = st.button("Clear", use_container_width=True, type="secondary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close input-container
    
    # Handle send button
    if send_button and user_input.strip():
        exchanges = len(st.session_state.investment_chat_history) // 2
        if exchanges >= 5:
            st.error('Conversation limit reached (max 5 exchanges). Start a new chat to continue.')
        else:
            # Authentication & credits check
            if 'user' not in st.session_state or not st.session_state.get('user'):
                st.error('You must be logged in to use this feature.')
            else:
                user = st.session_state.get('user')
                uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                credits = get_credits(uid) if uid else None
                COST = 100
                if credits is None or credits < COST:
                    st.error(f'Insufficient credits. {COST} required to use this feature.')
                else:
                    ok, new_or_msg = deduct_credits(uid, COST)
                    if not ok:
                        st.error(f'Could not deduct credits: {new_or_msg}')
                    else:
                        # Add user message
                        st.session_state.investment_chat_history.append({
                            "role": "user",
                            "content": user_input
                        })
                        # Get user's investment profile if configured
                        user_profile = None
                        if "investment_style" in st.session_state and "risk_tolerance" in st.session_state:
                            user_profile = f"{st.session_state.get('investment_style', '')}, {st.session_state.get('risk_tolerance', '')}"
                        # Generate AI response with conversation context
                        try:
                            with st.spinner("🤔 Thinking..."):
                                # If we have a Finviz snapshot loaded and the chat helper is available,
                                # prefer that as it injects the snapshot as system context.
                                snapshot = st.session_state.get('investment_snapshot', {})
                                if snapshot and 'scripts' in globals() and 'finviz_chat' in globals():
                                    # attempt to import the helper at runtime
                                    try:
                                        from scripts.finviz_chat import chat_with_snapshot
                                        ai_response = chat_with_snapshot(
                                            st.session_state.investment_chat_history[:-1],
                                            snapshot,
                                            st.session_state.get('investment_ticker', ''),
                                            api_key=os.getenv('OPENAI_API_KEY')
                                        )
                                    except Exception:
                                        # fallback to original generator
                                        ai_response = generate_investment_advice(
                                            user_input,
                                            user_profile,
                                            st.session_state.investment_chat_history[:-1]
                                        )
                                else:
                                    ai_response = generate_investment_advice(
                                        user_input,
                                        user_profile,
                                        st.session_state.investment_chat_history[:-1]
                                    )
                            # Add AI response
                            st.session_state.investment_chat_history.append({
                                "role": "assistant", 
                                "content": ai_response
                            })
                            # Clear input and refresh
                            st.experimental_rerun()
                        except Exception:
                            # rollback credits on failure
                            try:
                                supabase.table('users').update({'credits': new_or_msg}).eq('id', uid).execute()
                            except Exception:
                                pass
                            st.error('Error during analysis. Credits refunded.')
    
    # Handle clear button
    if clear_button:
        st.session_state.investment_chat_history = []
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Investment profile selection
    with st.expander("⚙️ Configure Investment Profile", expanded=False):
        st.markdown("""
        <p style="color: #ccc; margin-bottom: 1rem;">
        Configure your investment preferences for more personalized recommendations:
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment_style = st.selectbox(
                "Investment Style:",
                ["", "Value Investing", "Growth Investing", "Dividend Income", "Passive Investing", "Active Trading"],
                help="Choose your preferred investment approach",
                key="investment_style_selector"
            )
        
        with col2:
            risk_tolerance = st.selectbox(
                "Risk Tolerance:",
                ["", "Conservative", "Moderate", "Aggressive"],
                help="Select your risk comfort level",
                key="risk_tolerance_selector"
            )
        
        # Save preferences to session state
        if investment_style:
            st.session_state.investment_style = investment_style
        if risk_tolerance:
            st.session_state.risk_tolerance = risk_tolerance
        
        # Show current profile if configured
        if investment_style or risk_tolerance:
            st.success(f"✅ Profile: {investment_style} | Risk: {risk_tolerance}")
        
        # Show saved profile information
        if hasattr(st.session_state, 'investment_style') or hasattr(st.session_state, 'risk_tolerance'):
            saved_style = getattr(st.session_state, 'investment_style', 'Not set')
            saved_risk = getattr(st.session_state, 'risk_tolerance', 'Not set')
            st.info(f"🔹 **Saved Profile**: {saved_style} | {saved_risk}")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid #404040; color: #666;">
        <p style="margin: 0; font-size: 0.9rem;">
            💡 AI-powered investment assistant • Finviz screening • Put options guidance
        </p>
    </div>
    """, unsafe_allow_html=True)

def test_firecrawl_connection():
    """
    Prueba la conexión con Firecrawl
    """
    try:
        app = FirecrawlApp(api_key="fc-6d4cb3a2546c47c38f51a664e19c9216")
        
        # Probar con una URL simple usando la sintaxis correcta
        test_result = app.scrape_url("https://example.com")
        
        if test_result:
            return True, "Firecrawl connection successful"
        else:
            return False, "Firecrawl returned empty result"
            
    except Exception as e:
        return False, f"Firecrawl connection failed: {str(e)}"

def extract_content_with_firecrawl(url):
    """
    Extrae contenido usando Firecrawl como scraper alternativo con fallback de múltiples API keys
    """
    # Array de API keys para fallback automático
    api_keys = [
        "fc-6d4cb3a2546c47c38f51a664e19c9216",  # API key principal
        "fc-7f617b32f2734aa7aac035dbedd86270",  # Fallback 1
        "fc-6539c9ee9c0c4fab84933070cefa2cf4",  # Fallback 2
        "fc-b2241a8561bd4ff5a2d87916a52d7ce6",  # Fallback 3
        "fc-0c90998e15a14cd38f5ca4e73c5d3c30"   # Fallback 4
    ]
    
    last_error = None
    
    # Intentar con cada API key hasta que una funcione
    for i, api_key in enumerate(api_keys):
        try:
            # Inicializar Firecrawl con la API key actual
            app = FirecrawlApp(api_key=api_key)
            
            # Usar la sintaxis correcta de Firecrawl sin parámetros params
            scrape_result = app.scrape_url(url)
            
            if scrape_result:
                # Intentar obtener contenido de diferentes campos posibles
                content = None
                
                # Probar diferentes campos que puede retornar Firecrawl
                if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
                    content = scrape_result.markdown
                elif hasattr(scrape_result, 'content') and scrape_result.content:
                    content = scrape_result.content
                elif hasattr(scrape_result, 'text') and scrape_result.text:
                    content = scrape_result.text
                elif isinstance(scrape_result, dict):
                    # Si es un diccionario, intentar diferentes claves
                    content = (scrape_result.get('markdown') or 
                              scrape_result.get('content') or 
                              scrape_result.get('text') or
                              scrape_result.get('data', {}).get('markdown') or
                              scrape_result.get('data', {}).get('content'))
                elif isinstance(scrape_result, str):
                    content = scrape_result
                
                if content and len(str(content)) > 50:
                    # Limpiar el contenido
                    content = str(content)
                    content = re.sub(r'[#*`\[\]_~]', '', content)  # Remover markdown syntax
                    content = re.sub(r'\s+', ' ', content).strip()   # Normalizar espacios
                    content = content.replace('\n', ' ')  # Remover saltos de línea
                    
                    if len(content) > 100:
                        # Éxito - mostrar qué API key funcionó si no fue la primera
                        if i > 0:
                            print(f"✅ Firecrawl fallback successful using API key {i+1}/{len(api_keys)}")
                        return content[:3000], True  # Limitar a 3000 caracteres
                    else:
                        last_error = "Content too short via Firecrawl"
                        continue  # Intentar con la siguiente API key
                else:
                    last_error = f"No content found via Firecrawl. Result type: {type(scrape_result)}"
                    continue  # Intentar con la siguiente API key
            else:
                last_error = "No response from Firecrawl"
                continue  # Intentar con la siguiente API key
                
        except Exception as e:
            last_error = f"Firecrawl error with API key {i+1}: {str(e)}"
            # Si el error indica límite de rate o tokens, intentar siguiente key
            if any(keyword in str(e).lower() for keyword in ['rate limit', 'quota', 'credits', 'limit exceeded', 'unauthorized']):
                print(f"⚠️ API key {i+1} limit reached, trying fallback...")
                continue
            # Para otros errores, también intentar la siguiente key
            continue
    
    # Si llegamos aquí, todas las API keys fallaron
    return f"All Firecrawl API keys failed. Last error: {last_error}", False

def extract_article_content_auto(url):
    """
    Extrae contenido usando BeautifulSoup primero, Firecrawl como fallback
    Retorna: (content, success, method_used)
    """
    # Intentar primero con BeautifulSoup (más rápido)
    content, success = extract_content_with_beautifulsoup(url)
    
    if success:
        return content, True, "BeautifulSoup"
    
    # Si BeautifulSoup falla, intentar con Firecrawl silenciosamente
    content_fc, success_fc = extract_content_with_firecrawl(url)
    
    return content_fc, success_fc, "Firecrawl"

def extract_article_content(url, method="auto"):
    """
    Extrae el contenido completo de un artículo usando el método seleccionado
    method: "firecrawl" o "beautifulsoup"
    """
    if method == "firecrawl":
        # Usar Firecrawl
        return extract_content_with_firecrawl(url)
    elif method == "beautifulsoup":
        # Usar BeautifulSoup
        return extract_content_with_beautifulsoup(url)
    else:
        # Default a Firecrawl
        return extract_content_with_firecrawl(url)

def extract_content_with_beautifulsoup(url):
    """
    Extrae contenido usando BeautifulSoup (método clásico)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    alternative_headers = [
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        },
        {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Usar headers alternativos en intentos posteriores
            current_headers = headers if attempt == 0 else alternative_headers[attempt % len(alternative_headers)]
            
            response = requests.get(url, headers=current_headers, timeout=10)
            
            if response.status_code == 403:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return "Access denied (403). Using title-only analysis.", False
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Detectar si es un press release pagado
            page_text = soup.get_text().lower()
            if 'paid press release' in page_text or 'sponsored content' in page_text:
                return "This is a paid press release. Using title-only analysis.", False
            
            # Remover elementos no deseados
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Buscar contenido principal
            content_selectors = [
                'article', '.article-content', '.post-content', '.story-body',
                '.content', '.entry-content', '.article-body', 'main',
                '[class*="content"]', '[class*="article"]', '[class*="story"]'
            ]
            
            article_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text().strip()
                        if len(text) > len(article_content):
                            article_content = text
                    if len(article_content) > 200:
                        break
            
            # Si no encuentra contenido específico, usar todo el texto
            if len(article_content) < 200:
                article_content = soup.get_text()
            
            # Limpiar el contenido
            article_content = re.sub(r'\s+', ' ', article_content).strip()
            
            # Verificar si el contenido es sustancial
            if len(article_content) > 200:
                return article_content[:2000], True  # Limitar a 2000 caracteres
            else:
                return "Content too short. Using title-only analysis.", False
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return f"Error loading article: {str(e)}", False
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return f"Error processing article: {str(e)}", False
    
    return "Failed to extract content after all retries.", False

def sentimentAnalysis(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        try:
            url = finviz_url + ticker
            req = Request(url=url, headers={
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response = urlopen(req)
            html = BeautifulSoup(response, 'html.parser')
            news_table = html.find(id='news-table')
            
            if news_table:
                news_tables[ticker] = news_table
            else:
                st.warning(f"⚠️ No news found for {ticker}. This ticker might not exist or have recent news.")
                continue
                
        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
            continue

    if not news_tables:
        st.error("❌ No valid tickers found or no news available for the provided symbols.")
        return pd.DataFrame()

    parsed_data = []

    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            if row.find('a'):
                title = row.a.text
                link = 'https://finviz.com' + row.a['href'] if row.a.get('href', '').startswith('/') else row.a.get('href', '')
                
                date_data = row.td.text.replace("\r\n", "").split(' ')
                
                if len(date_data) == 21:
                    article_time = date_data[12]
                    date = "Today"
                else:
                    date = date_data[12]
                    article_time = date_data[13] if len(date_data) > 13 else "N/A"
                
                parsed_data.append([ticker, date, article_time, title, link])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'link'])
    
    # Extraer contenido completo de los artículos
    full_contents = []
    extraction_success = []
    methods_used = []
    
    # Show a single progress bar (percentage) so the user sees extraction progress
    progress_bar = st.progress(0)
    for idx, row in df.iterrows():
        
        if row['link'] and row['link'] != '':
            # Usar método automático: BeautifulSoup → Firecrawl
            content, success, method = extract_article_content_auto(row['link'])
            
            full_contents.append(content)
            extraction_success.append(success)
            methods_used.append(method)
        else:
            full_contents.append("No link available")
            extraction_success.append(False)
            methods_used.append("N/A")
        
        time.sleep(0.1)  # Pequeña pausa para evitar sobrecargar el servidor
        # Update progress (percentage)
        try:
            pct = int(((idx + 1) / len(df)) * 100)
            progress_bar.progress(pct)
        except Exception:
            pass
    
    df['full_content'] = full_contents
    df['extraction_success'] = extraction_success
    df['method_used'] = methods_used
    
    # finalize and remove progress bar
    try:
        progress_bar.progress(100)
        progress_bar.empty()
    except Exception:
        pass
    
    # SECCIÓN: ESTADÍSTICAS DE EXTRACCIÓN
    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
    # Compute extraction stats (not displayed here)
    success_count = sum(extraction_success)
    method_stats = {}
    for method in methods_used:
        method_stats[method] = method_stats.get(method, 0) + 1

    if success_count == 0:
        st.error("❌ Could not extract content from any article. Please verify the URLs.")
        return
    # Separador (visual)
    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

    # Análisis de sentimiento
    vader = SentimentIntensityAnalyzer()

    def analyze_sentiment(row):
        # Si la extracción fue exitosa, analizar el contenido completo
        if row['extraction_success'] and len(row['full_content']) > 50:
            text_to_analyze = f"{row['title']} {row['full_content']}"
        else:
            # Fallback al título solamente
            text_to_analyze = row['title']
        
        return vader.polarity_scores(text_to_analyze)['compound']

    df['compound'] = df.apply(analyze_sentiment, axis=1)
    
    # Procesar fechas
    df.loc[df['date'] == "Today", 'date'] = date2.today()
    df['date'] = pd.to_datetime(df.date).dt.date

    return df

def create_sentiment_gauge(sentiment_score, ticker):
    """
    Crea un gauge chart para mostrar el sentimiento
    """
    # Convertir el score de -1,1 a 0,100 para el gauge
    gauge_value = (sentiment_score + 1) * 50
    
    # Determinar color basado en el sentimiento
    if sentiment_score >= 0.1:
        color = "green"
        sentiment_text = "Positive"
    elif sentiment_score <= -0.1:
        color = "red"
        sentiment_text = "Negative"
    else:
        color = "yellow"
        sentiment_text = "Neutral"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
        'text': f"Sentiment for {ticker}",
            'font': {'color': 'white', 'size': 16}
        },
        delta = {
            'reference': 50,
            'font': {'color': 'white'}
        },
        gauge = {
            'axis': {
                'range': [None, 100],
                'tickcolor': 'white',
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "rgba(255,0,0,0.2)"},
                {'range': [30, 70], 'color': "rgba(255,255,0,0.2)"},
                {'range': [70, 100], 'color': "rgba(0,255,0,0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 90
            }
        },
        number = {'font': {'color': 'white', 'size': 20}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    
    return fig, sentiment_text


def show_finviz_dashboard_chat():
    """Embed the Finviz dashboard + chat (ported from finviz_test.py)"""
    # Hide Streamlit top menu/header/footer for this page only
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header[data-testid='stHeader'] {visibility: hidden;}
        section[role='banner'] {display: none;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title in white and updated text
    st.markdown('<h1 style="color:#ffffff; margin-bottom: 0.25rem;">KPI Dashboard & Chat With AI</h1>', unsafe_allow_html=True)

    # Description in strong green (KPI analysis only)
    st.markdown('<p style="color:#10a37f; margin-top: 0.1rem;">Analyze and visualize key performance indicators (KPIs) in a beautiful dashboard, then use the chat below to explore and discuss the data.</p>', unsafe_allow_html=True)

    from scripts.scrape_finviz import scrape_finviz_snapshot
    from scripts.finviz_chat import chat_with_snapshot

    # ticker label (rendered green) and input (hide default label)
    st.markdown('<div style="color:#10a37f; font-weight:700;">Ticker</div>', unsafe_allow_html=True)
    ticker = st.text_input('', value=st.session_state.get('finviz_ticker', 'A'), key='finviz_ticker', label_visibility='collapsed')

    col1, col2 = st.columns([2, 1])

    # Automatic fetch when ticker changes or when no snapshot loaded
    with col1:
        snapshot = st.session_state.get('snapshot', {})
        last = st.session_state.get('last_finviz_ticker')
        current = st.session_state.get('finviz_ticker')
        if current and current != last:
            # Use a silent spinner without the ticker message
            with st.spinner(''):
                try:
                    snapshot, ok = scrape_finviz_snapshot(current)
                    if ok:
                        st.session_state['snapshot'] = snapshot
                        st.session_state['last_finviz_ticker'] = current
                        # store the date the snapshot was collected (formatted in English)
                        try:
                            st.session_state['snapshot_date'] = date2.today().strftime('%B %d, %Y')
                        except Exception:
                            st.session_state['snapshot_date'] = None
                    else:
                        st.session_state['snapshot'] = {}
                        st.session_state['last_finviz_ticker'] = current
                        st.error('Snapshot table not found or page structure changed.')
                except Exception as e:
                    st.session_state['snapshot'] = {}
                    st.session_state['last_finviz_ticker'] = current
                    st.exception(e)

        snapshot = st.session_state.get('snapshot', {})

        if snapshot:
            # Small map of human-friendly KPI descriptions shown on hover
            KPI_DESCRIPTIONS = {
                'Market Cap': 'Total market value of the company\'s outstanding shares.',
                'P/E': 'Price-to-Earnings ratio: price divided by earnings per share.',
                'EPS (ttm)': 'Earnings per share (trailing twelve months).',
                'Dividend %': 'Annual dividend yield as a percentage of current price.',
                'Volume': 'Number of shares traded during the latest session.',
                'Avg Volume': 'Average number of shares traded per day.',
                'RSI (14)': 'Relative Strength Index (14-day) — momentum indicator.',
            }

            # Inject CSS for KPI cards: 3D hover, subtle float animation, and hover description
            st.markdown("""
            <style>
            .kpi-grid { display:flex; flex-wrap:wrap; gap:16px; margin-top:12px; }
            .kpi-card {
                min-width:210px; flex:1 1 210px;
                /* darker, slightly textured tile to read well on dark backgrounds */
                background: linear-gradient(180deg, #121212 0%, #0e0e0e 100%);
                padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.04);
                box-shadow: 0 6px 18px rgba(2,6,23,0.6); transform-style: preserve-3d;
                transition: transform 0.35s cubic-bezier(.2,.8,.2,1), box-shadow 0.35s ease;
                will-change: transform; backface-visibility: hidden;
                perspective: 800px;
                animation: floaty 6s ease-in-out infinite; position:relative; overflow:visible;
            }
            .kpi-card:hover {
                transform: translateY(-8px) rotateX(6deg) rotateY(-6deg) scale(1.02);
                box-shadow: 0 18px 40px rgba(2,6,23,0.75);
            }
            /* subtle grey dot (viñeta) before the KPI title */
            .kpi-key { color: #ffffff; font-weight:800; margin-bottom:6px; transform: translateZ(12px); }
            .kpi-key::before { content: ''; display:inline-block; width:10px; height:10px; background:#6b6b6b; border-radius:50%; margin-right:8px; vertical-align:middle; }
            .kpi-value { color: #ffffff; font-size:1.05rem; font-weight:600; transform: translateZ(6px); }
            .kpi-desc { color: rgba(255,255,255,0.88); font-size:0.85rem; margin-top:8px; opacity:0; transition: opacity 180ms ease, transform 180ms ease; transform: translateY(6px); }
            .kpi-card:hover .kpi-desc { opacity:1; transform: translateY(0); }

            @keyframes floaty {
                0% { transform: translateY(0) rotateX(0) rotateY(0); }
                50% { transform: translateY(-6px) rotateX(1deg) rotateY(-1deg); }
                100% { transform: translateY(0) rotateX(0) rotateY(0); }
            }
            </style>
            """, unsafe_allow_html=True)

            # Dashboard heading with ticker in green; include collection date when available
            snapshot_date = st.session_state.get('snapshot_date')
            date_suffix = f" — Collected on {snapshot_date}" if snapshot_date else ""
            st.markdown(f'<h3 style="color:#10a37f">Dashboard for {current or ticker}{date_suffix}</h3>', unsafe_allow_html=True)

            items = list(snapshot.items())
            # Render key/value pairs as a responsive grid with green keys and hover descriptions
            html_parts = ['<div class="kpi-grid">']
            for key, val in items:
                desc = KPI_DESCRIPTIONS.get(key, '')
                desc_html = f"<div class='kpi-desc'>{desc}</div>" if desc else ''
                # Use title attribute as a small accessibility fallback
                safe_title = desc.replace("'", "\'") if desc else ''
                html_parts.append(
                    f"<div class='kpi-card' title=\'{safe_title}\'>"
                    + f"<div class='kpi-key'>{key}</div>"
                    + f"<div class='kpi-value'>{val}</div>"
                    + desc_html
                    + "</div>"
                )
            html_parts.append('</div>')
            st.markdown(''.join(html_parts), unsafe_allow_html=True)

            # raw JSON and success message removed — KPIs displayed above only
        else:
            st.info('No snapshot loaded. Enter a ticker.')

    with col2:
        st.markdown('<div style="color:#10a37f; font-weight:800; font-size:1.15rem; margin-bottom:6px;">Chat with AI</div>', unsafe_allow_html=True)
        # Show user's credits (if available)
        try:
            credits_val = None
            if 'user' in st.session_state and st.session_state.get('user'):
                user_obj = st.session_state.get('user')
                uid = user_obj.get('id') if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
                credits_val = get_credits(uid) if uid else None
            credits_html = f"<div id='credits-bubble-kpis' style='background:linear-gradient(90deg,#0ea37f,#076b57);padding:6px 10px;border-radius:999px;color:#fff;font-weight:700;box-shadow:0 4px 12px rgba(6,45,40,0.25);font-size:0.95rem;margin-bottom:8px;'>Credits: {credits_val if credits_val is not None else 'N/A'}</div>"
            st.markdown(credits_html, unsafe_allow_html=True)
        except Exception:
            # non-fatal: if credits retrieval fails, show N/A
            st.markdown("<div style='color:#bbb; margin-bottom:8px;'>Credits: N/A</div>", unsafe_allow_html=True)
        # Use a dedicated session key for KPI chat so context persists separately
        if 'kpis_chat_history' not in st.session_state:
            st.session_state['kpis_chat_history'] = []

        st.markdown('<div style="color:#10a37f; font-weight:700; margin-top:6px;">Your question</div>', unsafe_allow_html=True)
        chat_input = st.text_area('', height=120, key='kpis_chat_input', label_visibility='collapsed')

        if st.button('Send'):
            if not chat_input.strip():
                st.warning('Write a question first')
            else:
                # enforce max 5 user messages (not exchanges)
                user_msgs = [m for m in st.session_state.get('kpis_chat_history', []) if m.get('role') == 'user']
                if len(user_msgs) >= 5:
                    st.error('Conversation limit reached (max 5 user messages). Start a new chat to continue.')
                    st.stop()
                else:
                    # Authentication & credits check (require login and 30 credits to start)
                    if 'user' not in st.session_state or not st.session_state.get('user'):
                        st.error('You must be logged in to use this feature.')
                    else:
                        user = st.session_state.get('user')
                        uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                        credits = get_credits(uid) if uid else None
                        COST = 30
                        if credits is None or credits < COST:
                            st.error(f'Insufficient credits. {COST} required to use this feature.')
                        else:
                            # save previous credits for rollback
                            prev_credits = credits
                            ok, new_or_msg = deduct_credits(uid, COST)
                            if not ok:
                                st.error(f'Could not deduct credits: {new_or_msg}')
                            else:
                                # add user message and call the snapshot-aware chat helper using prior context
                                st.session_state['kpis_chat_history'].append({'role': 'user', 'content': chat_input})
                                try:
                                    snapshot = st.session_state.get('snapshot', {})
                                    # pass prior messages (exclude current) as context, mirroring Investment
                                    ai_response = chat_with_snapshot(
                                        st.session_state['kpis_chat_history'],
                                        snapshot,
                                        current or ticker,
                                        api_key=os.getenv('OPENAI_API_KEY')
                                    )
                                    st.session_state['kpis_chat_history'].append({'role': 'assistant', 'content': ai_response})
                                    st.experimental_rerun()
                                except Exception as e:
                                    # rollback credits on failure
                                    try:
                                        supabase.table('users').update({'credits': prev_credits}).eq('id', uid).execute()
                                    except Exception:
                                        pass
                                    st.error('Error during analysis. Credits refunded.')

        for msg in st.session_state['kpis_chat_history']:
            if msg['role'] == 'user':
                st.markdown(f"<div style='background:rgba(255,255,255,0.02);padding:12px;border-radius:8px;margin:8px 0;'><strong style='color:#10a37f;'>You:</strong> <span style='color:#fff'>{msg['content']}</span></div>", unsafe_allow_html=True)
            else:
                # Assistant label in green, response text in white
                st.markdown(f"<div style='background:rgba(16,163,127,0.06);padding:12px;border-radius:8px;margin:8px 0;'><strong style='color:#10a37f;'>Assistant:</strong> <span style='color:#ffffff'>{msg['content']}</span></div>", unsafe_allow_html=True)

        if st.button('Clear chat'):
            st.session_state['kpis_chat_history'] = []

    # warning removed per user request

def show_page():
    # CSS personalizado para modo oscuro y diseño moderno
    st.markdown("""
    <style>
    /* Dark mode background */
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c1427 0%, #12203e 100%);
        color: white !important;
    }
    
    /* Asegurar que todo el texto sea blanco */
    * {
        color: white !important;
    }
    
    /* Específico para elementos de Streamlit */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stText, p, span, div {
        color: white !important;
    }
    
    /* Subheaders específicos */
    .css-10trblm, .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3 {
        color: white !important;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    /* Cards with glassmorphism effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        width: 100%;
        max-width: none;
        box-sizing: border-box;
    }
    
    /* Secciones principales con ancho completo */
    .full-width-section {
        width: 100% !important;
        max-width: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Headers con ancho completo */
    .section-header {
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Custom metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        color: black !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Text input labels */
    .stTextInput > label {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Success/Warning/Error messages */
    .stAlert {
        color: white !important;
    }
    
    /* Logo animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .logo {
        animation: float 3s ease-in-out infinite;
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal con logo
    st.markdown("""
    <div class="main-header">
        <div class="logo">📈</div>
        <h1 style="margin: 0; font-size: 3rem; color: white; font-weight: 700;">
           Stock Sentiment Analyzer
        </h1>
        <p style="font-size: 1.3rem; opacity: 0.9; margin-top: 1rem; color: #00d4ff;">
           Automated sentiment analysis with AI for intelligent investment decisions
        </p>
        <p style="font-size: 1rem; opacity: 0.7; margin-top: 0.5rem;">
           🤖 Smart scraping • Real-time analysis • 🎯 Actionable insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input personalizado para tickers
    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Analysis Configuration</h3>
        <p>Enter the stock symbols you want to analyze, separated by commas</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Examples: AAPL, MSFT, GOOGL, TSLA, NVDA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input de usuario
    user_input = st.text_input(
        "Stock symbols:",
        value="",
        help="Enter symbols separated by commas",
        label_visibility="collapsed"
    )
    
    # Información amigable para el cliente sobre la extracción y el análisis
    st.markdown("""
    <div class="glass-card">
        <h3>🤖 AI-Powered News Analysis</h3>
    <p>We analyze market news and sentiment to quickly deliver relevant, actionable insights. Our system automatically collects and processes articles, highlighting key statistics, important headlines, and signals to watch — all presented clearly to make decision‑making easier.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Procesar input
    if user_input:
        # Limpiar y validar tickers
        raw_tickers = [ticker.strip().upper() for ticker in user_input.split(',') if ticker.strip()]
        
        # Filtrar solo símbolos válidos (letras únicamente, 1-5 caracteres)
        tickers = []
        invalid_tickers = []
        
        for ticker in raw_tickers:
            # Remover caracteres no alfabéticos
            clean_ticker = ''.join(c for c in ticker if c.isalpha())
            
            if len(clean_ticker) >= 1 and len(clean_ticker) <= 5:
                tickers.append(clean_ticker)
            else:
                invalid_tickers.append(ticker)
        
        # Mostrar advertencias para tickers inválidos
        if invalid_tickers:
            st.warning(f"⚠️ Invalid symbols ignored: {', '.join(invalid_tickers)}")
        
        if not tickers:
            st.error("❌ No valid symbols found. Please enter valid stock symbols (e.g., AAPL, MSFT)")
            return
        
        # --- Authentication & Credits UI ---
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            # Determine whether we have an authenticated user with a valid id
            current_user = st.session_state.get('user') if 'user' in st.session_state else None
            current_uid = None
            if current_user:
                try:
                    current_uid = current_user.get('id') if isinstance(current_user, dict) else getattr(current_user, 'id', None)
                except Exception:
                    current_uid = None

            # If no valid UID, show login/register forms
            if not current_uid:
                st.markdown("#### 🔐 Login or Register")
                email = st.text_input('Email', key='auth_email')
                password = st.text_input('Password', type='password', key='auth_password')
                # Remember on this device is mandatory
                remember_me = True
                coll1, coll2 = st.columns(2)
                with coll1:
                    if st.button('Login'):
                        ok, resp = sign_in(email, password)
                        if ok:
                            # Normalize response to a user object that contains 'id' and 'email'
                            user_obj = None
                            if isinstance(resp, dict):
                                user_obj = resp.get('user') or resp.get('data', {}).get('user') or resp.get('data', {}).get('session', {}).get('user') or resp
                            else:
                                user_obj = getattr(resp, 'user', None) or getattr(resp, 'data', None) or resp
                            # set session and persist locally (remember is mandatory)
                            st.session_state['user'] = user_obj
                            try:
                                save_local_session(user_obj)
                                st.session_state['remember_me'] = True
                            except Exception:
                                pass
                            # ensure we have the uid before saving refresh token
                            try:
                                uid = user_obj.get('id') if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
                            except Exception:
                                uid = None
                            # If the response contains a refresh token, save it server-side in Supabase and add remember_uid redirect
                            try:
                                refresh_token = None
                                if isinstance(resp, dict):
                                    refresh_token = resp.get('session', {}).get('refresh_token') or resp.get('refresh_token') or resp.get('data', {}).get('refresh_token')
                                else:
                                    refresh_token = getattr(resp, 'refresh_token', None) or (getattr(resp, 'data', None) and getattr(getattr(resp, 'data', None), 'refresh_token', None))
                                if refresh_token and uid:
                                    save_refresh_token_to_db(uid, refresh_token)
                                    # always append remember_uid so reloads can restore session
                                    js = f"<script>window.location.search = '?remember_uid={uid}';</script>"
                                    st.markdown(js, unsafe_allow_html=True)
                            except Exception:
                                pass
                            # ensure user record with credits exists
                            try:
                                uid = user_obj.get('id') if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
                                create_user_record_if_missing(uid, email)
                                st.success('Logged in')
                                # Refresh so the UI switches to the compact session bar
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                            except Exception:
                                st.success('Logged in (no DB write)')
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                        else:
                            st.error(f'Login failed: {resp}')
                with coll2:
                    if st.button('Register'):
                        ok, resp = sign_up(email, password)
                        if ok:
                                # sign_up may return a dict with 'user' or a user object
                                user_obj = None
                                if isinstance(resp, dict):
                                    user_obj = resp.get('user') or resp
                                else:
                                    user_obj = getattr(resp, 'user', None) or resp

                                # Check common confirmation fields set by Supabase
                                confirmed = False
                                try:
                                    if isinstance(user_obj, dict):
                                        confirmed = bool(user_obj.get('email_confirmed_at') or user_obj.get('confirmed_at'))
                                    else:
                                        confirmed = bool(getattr(user_obj, 'email_confirmed_at', None) or getattr(user_obj, 'confirmed_at', None))
                                except Exception:
                                    confirmed = False

                                if confirmed:
                                    # user already confirmed: set session and create DB record with credits
                                    st.session_state['user'] = user_obj
                                    # Persist session locally if requested
                                    try:
                                        if st.session_state.get('remember_me'):
                                            save_local_session(user_obj)
                                    except Exception:
                                        pass
                                    # Save refresh token if present
                                    try:
                                        refresh_token = None
                                        if isinstance(resp, dict):
                                            refresh_token = resp.get('session', {}).get('refresh_token') or resp.get('refresh_token') or resp.get('data', {}).get('refresh_token')
                                        else:
                                            refresh_token = getattr(resp, 'refresh_token', None) or getattr(resp, 'data', None) and getattr(getattr(resp, 'data', None), 'refresh_token', None)
                                        if refresh_token and uid:
                                            save_refresh_token_to_db(uid, refresh_token)
                                            if st.session_state.get('remember_me'):
                                                js = f"<script>window.location.search = '?remember_uid={uid}';</script>"
                                                st.markdown(js, unsafe_allow_html=True)
                                    except Exception:
                                        pass
                                    uid = user_obj.get('id') if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
                                    create_user_record_if_missing(uid, email)
                                    st.success('Registered and logged in')
                                    # Immediately rerun so the logged-in bar replaces the form
                                    try:
                                        st.experimental_rerun()
                                    except Exception:
                                        pass
                                else:
                                    # Not confirmed yet: do not set session; let Supabase handle confirmation
                                    st.success('Registration received')
                                    st.info('A confirmation email has been sent by Supabase. Please confirm your email to continue.')
                                    st.warning('You must confirm your email to receive the 300 credits.')
                        else:
                            st.error(f'Registration failed: {resp}')
            else:
                # Compact session bar when the user is logged in
                user = st.session_state.get('user')
                uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                email = user.get('email') if isinstance(user, dict) else getattr(user, 'email', '')
                credits = get_credits(uid) if uid else None

                # nicer session bar with spacing and refresh controls
                bar_col = st.columns([1])[0]
                with bar_col:
                    # ensure we fetch latest credits when rendering
                    try:
                        credits = get_credits(uid) if uid else credits
                    except Exception:
                        pass

                    st.markdown(f"""
                    <div style='background:#072227;padding:14px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.04);box-shadow:0 6px 18px rgba(2,12,10,0.4); margin-bottom:18px;'>
                        <div style='display:flex;align-items:center;gap:14px;'>
                            <div style='flex:1'>
                                <strong style='color:#10a37f;font-size:1rem;'>Session started with</strong>
                                <span style='color:#fff; margin-left:10px; font-weight:700; font-size:1rem;'>{email}</span>
                            </div>
                            <div style='display:flex;align-items:center;gap:10px;'>
                                <div id='credits-bubble' style='background:linear-gradient(90deg,#0ea37f,#076b57);padding:8px 12px;border-radius:999px;color:#fff;font-weight:700;box-shadow:0 4px 12px rgba(6,45,40,0.4);'>Credits: {credits if credits is not None else 'N/A'}</div>
                                <div id='credits-refresh-area'></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Inject a lightweight JS snippet that updates only the credits bubble every 10s
                try:
                    # Only inject if we have a valid uid and SUPABASE_URL / SUPABASE_KEY are set
                    if uid and SUPABASE_URL and SUPABASE_KEY:
                        safe_url = SUPABASE_URL.rstrip('/')
                        # Use the public anon key for read-only select (ensure this is anon, not service_role)
                        js_fetch = f"""
                        <script>
                        (function(){{
                            const uid = '{uid}';
                            const url = '{safe_url}/rest/v1/users?id=eq.' + uid + '&select=credits';
                            const headers = {{ 'apikey': '{SUPABASE_KEY}', 'Authorization': 'Bearer {SUPABASE_KEY}' }};

                            async function updateCredits(){{
                                try{{
                                    const r = await fetch(url, {{ method: 'GET', headers: headers }});
                                    if(!r.ok) return;
                                    const data = await r.json();
                                    if(Array.isArray(data) && data.length>0){{
                                        const creds = data[0].credits;
                                        const el = document.getElementById('credits-bubble');
                                        if(el) el.innerText = 'Credits: ' + creds;
                                        const invEl = document.getElementById('credits-bubble-investment');
                                        if(invEl) invEl.innerText = 'Credits: ' + creds;
                                    }}
                                }}catch(e){{
                                    // ignore
                                }}
                            }}

                            // Initial update and periodic refresh
                            updateCredits();
                            if(!window._credits_interval) window._credits_interval = setInterval(updateCredits, 10000);
                        }})();
                        </script>
                        """
                        st.markdown(js_fetch, unsafe_allow_html=True)
                except Exception:
                    pass
                # Logout button removed by user request

        # Start Analysis button (requires login & credits)
        with col_center:
            if st.button("🚀 Start Analysis", use_container_width=True):
                # Authentication check
                if 'user' not in st.session_state or not st.session_state.get('user'):
                    st.error('You must be logged in to start an analysis. Please register or login.')
                    return
                user = st.session_state.get('user')
                uid = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)
                if not uid:
                    st.error('User ID not available. Please re-login.')
                    return
                # Check credits (50 per analysis)
                credits = get_credits(uid)
                if credits is None:
                    st.error('Could not retrieve credits. Please try again later.')
                    return
                COST = 50
                if credits < COST:
                    st.error(f'Insufficient credits. You have {credits} credits, but {COST} are required to start an analysis.')
                    return
                # Deduct credits
                ok, new_or_msg = deduct_credits(uid, COST)
                if not ok:
                    st.error(f'Could not deduct credits: {new_or_msg}')
                    return
                # Proceed with analysis
                if tickers:
                    with st.spinner('Analyzing news and sentiment...'):
                        df = sentimentAnalysis(tickers)
                    
                    # Verificar si obtuvimos datos
                    if df.empty:
                        st.error("❌ Could not obtain data for any of the provided symbols. Please verify that the symbols are valid.")
                        return
                    
                    # Mostrar resultados
                    st.markdown("""
                    <div class="glass-card">
                        <h2>Analysis Results</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métricas generales
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Positive News", len(df[df['compound'] > 0.1]))
                    with col2:
                        st.metric("Negative News", len(df[df['compound'] < -0.1]))
                    with col3:
                        st.metric("Neutral News", len(df[(df['compound'] >= -0.1) & (df['compound'] <= 0.1)]))
                    with col4:
                        st.metric("Total Articles", len(df))
                    
                    # Información sobre la calidad del análisis
                    st.subheader("Content Analysis Quality")

                    # Contar artículos con contenido exitosamente extraído
                    successful_extractions = len(df[df['extraction_success'] == True])
                    failed_extractions = len(df[df['extraction_success'] == False])
                    press_releases = len(df[df['full_content'].str.contains('paid press release', case=False, na=False)])
                    success_rate = (successful_extractions / len(df)) * 100 if len(df) > 0 else 0

                    # Use HTML cards to ensure labels are displayed fully and responsively
                    quality_col1, quality_col2, quality_col3 = st.columns([1,1,1])
                    with quality_col1:
                        st.markdown(f"""
                        <div style="background:#0b0b0b;padding:16px;border-radius:10px;border:1px solid #222;text-align:center;">
                            <div style="color:#fff;font-weight:700;font-size:0.95rem;margin-bottom:6px;">✅ Articles with full content</div>
                                <div style="color:#10a37f;font-size:1.25rem;font-weight:700;">{successful_extractions}/{len(df)}</div>
                                <div style="color:#9aa0a6;font-size:0.85rem;margin-top:6px;">Number of articles with full content</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with quality_col2:
                        st.markdown(f"""
                        <div style="background:#0b0b0b;padding:16px;border-radius:10px;border:1px solid #222;text-align:center;">
                            <div style="color:#fff;font-weight:700;font-size:0.95rem;margin-bottom:6px;">Full content rate</div>
                            <div style="color:#10a37f;font-size:1.25rem;font-weight:700;">{success_rate:.1f}%</div>
                            <div style="color:#9aa0a6;font-size:0.85rem;margin-top:6px;">Percentage of articles with full content</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with quality_col3:
                        st.markdown(f"""
                        <div style="background:#0b0b0b;padding:16px;border-radius:10px;border:1px solid #222;text-align:center;">
                            <div style="color:#fff;font-weight:700;font-size:0.95rem;margin-bottom:6px;">📰 Press Releases</div>
                            <div style="color:#10a37f;font-size:1.25rem;font-weight:700;">{press_releases}</div>
                            <div style="color:#9aa0a6;font-size:0.85rem;margin-top:6px;">Number of paid press releases (title-only)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mostrar insights sobre la calidad
                    if success_rate > 80:
                        st.success(f"✅ **High Quality Analysis**: {success_rate:.1f}% of articles analyzed with full content")
                    elif success_rate > 60:
                        st.warning(f"⚠️ **Good Analysis**: {success_rate:.1f}% of articles analyzed with full content. {failed_extractions} articles used title-only analysis.")
                    else:
                        st.error(f"⚠️ **Limited Analysis**: Only {success_rate:.1f}% of articles analyzed with full content. {failed_extractions} articles used title-only analysis.")
                    
                    # Análisis por ticker
                    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-header">Detailed Analysis by Stock</h2>', unsafe_allow_html=True)
                    
                    # Crear análisis individual para cada ticker
                    for ticker in tickers:
                        ticker_data = df[df['ticker'] == ticker]
                        
                        if len(ticker_data) > 0:
                            avg_sentiment = ticker_data['compound'].mean()
                            
                            # Card para cada acción con mejor estructura
                            st.markdown(f"""
                            <div class="glass-card" style="width: 100%; max-width: none;">
                                <h3 style="margin-bottom: 15px;">🏢 {ticker.upper()}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # GAUGE PRINCIPAL - Ancho completo y muy grande
                            st.markdown("### Sentiment Gauge")
                            fig, sentiment_text = create_sentiment_gauge(avg_sentiment, ticker)
                            # Gauge dominante en toda la pantalla
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            
                            # MÉTRICAS EN COLUMNAS DEBAJO DEL GAUGE
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                            with col_metrics1:
                                st.metric("Articles Analyzed", len(ticker_data))
                            with col_metrics2:
                                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                            with col_metrics3:
                                st.metric("Trend", sentiment_text)
                            
                            # AI ANALYSIS - Full width below
                            st.markdown("### 🤖 Artificial Intelligence Analysis")
                            
                            # Prepare data for AI
                            ticker_articles = ticker_data.to_dict('records')
                            content_data = []
                            
                            for article in ticker_articles:
                                content_data.append({
                                    'title': article['title'],
                                    'content': article['full_content'] if article['extraction_success'] else article['title']
                                })
                            
                            # AI analysis in full-width container
                            with st.container():
                                with st.spinner(f'Generating intelligent analysis for {ticker}...'):
                                    # compute some stats to pass to the AI
                                    stats = {
                                        'articles_count': len(ticker_data),
                                        'full_count': int(ticker_data['extraction_success'].sum()),
                                        'success_rate': (ticker_data['extraction_success'].sum() / len(ticker_data) * 100) if len(ticker_data)>0 else 0,
                                        'press_releases': int(ticker_data['full_content'].str.contains('paid press release', case=False, na=False).sum())
                                    }
                                    # simple sentiment sorting for top pos/neg headlines
                                    sorted_by_sent = ticker_data.sort_values('compound')
                                    top_neg = list(sorted_by_sent.head(3)['title'].astype(str))
                                    top_pos = list(sorted_by_sent.tail(3)['title'].astype(str))[::-1]

                                    ai_summary = generate_ai_summary(content_data, ticker, avg_sentiment, stats=stats, top_pos=top_pos, top_neg=top_neg)
                                # Display highlighted stats for the AI to reference (visible to user)
                                stats_html = f"""
                                <div style='display:flex; gap:12px; width:100%; margin-bottom:12px;'>
                                    <div style='background:#071018;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);flex:1;text-align:center;'>
                                        <div style='color:#9aa0a6;font-size:0.85rem;'>Articles</div>
                                        <div style='color:#10a37f;font-weight:700;font-size:1.1rem;'>{stats['articles_count']}</div>
                                    </div>
                                    <div style='background:#071018;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);flex:1;text-align:center;'>
                                        <div style='color:#9aa0a6;font-size:0.85rem;'>Full Content</div>
                                        <div style='color:#10a37f;font-weight:700;font-size:1.1rem;'>{stats['full_count']}</div>
                                    </div>
                                    <div style='background:#071018;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);flex:1;text-align:center;'>
                                        <div style='color:#9aa0a6;font-size:0.85rem;'>Success Rate</div>
                                        <div style='color:#10a37f;font-weight:700;font-size:1.1rem;'>{stats['success_rate']:.1f}%</div>
                                    </div>
                                    <div style='background:#071018;padding:12px;border-radius:8px;border:1px solid rgba(255,255,255,0.03);flex:1;text-align:center;'>
                                        <div style='color:#9aa0a6;font-size:0.85rem;'>Press Releases</div>
                                        <div style='color:#10a37f;font-weight:700;font-size:1.1rem;'>{stats['press_releases']}</div>
                                    </div>
                                </div>
                                """

                                st.markdown(stats_html, unsafe_allow_html=True)

                                # Compact AI box (scrollable) and an expander to view full analysis
                                st.markdown(f"<div style='max-height:220px; overflow:auto; padding-bottom:4px;'>{ai_summary}</div>", unsafe_allow_html=True)
                                with st.expander("Expand AI Analysis (show full)"):
                                    st.markdown(ai_summary, unsafe_allow_html=True)
                            
                            # Separator between stocks if not the last one
                            if ticker != tickers[-1]:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("---")
                                st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Cerrar sección de análisis detallado
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Separador visual
                    st.markdown("---")
                    
                    # SECCIÓN 2: DETALLES DE ARTÍCULOS (sección de análisis por acción eliminada)
                    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-header">📋 Article Details</h2>', unsafe_allow_html=True)
                    
                    # CSS específico para mejorar la tabla
                    st.markdown("""
                    <style>
                    /* Mejorar la visualización de la tabla */
                    .stDataFrame {
                        width: 100% !important;
                    }
                    .stDataFrame > div {
                        width: 100% !important;
                        overflow-x: auto !important;
                        scroll-behavior: smooth !important;
                    }
                    /* Hacer que las celdas de la tabla sean más legibles */
                    .stDataFrame table {
                        width: 100% !important;
                        min-width: 1200px !important;
                    }
                    .stDataFrame th, .stDataFrame td {
                        padding: 8px 12px !important;
                        text-align: left !important;
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                    }
                    /* Títulos de artículos más largos */
                    .stDataFrame td:nth-child(2) {
                        max-width: 400px !important;
                        white-space: normal !important;
                        word-wrap: break-word !important;
                    }
                    /* URLs más legibles y clicables */
                    .stDataFrame td:nth-child(3) {
                        max-width: 300px !important;
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                        color: #00d4ff !important;
                        text-decoration: underline !important;
                        cursor: pointer !important;
                    }
                    
                    .stDataFrame td:nth-child(3):hover {
                        color: #ffffff !important;
                        background-color: rgba(0, 212, 255, 0.2) !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Crear tabla más legible y organizada (sin columna "Scraper Used")
                    display_df = df[['ticker', 'title', 'link', 'compound', 'extraction_success']].copy()

                    # Acortar títulos muy largos para mejor visualización
                    display_df['title_short'] = display_df['title'].apply(
                        lambda x: x[:80] + "..." if len(x) > 80 else x
                    )

                    # Crear URLs limpias para la tabla
                    display_df['url_clean'] = display_df['link'].apply(
                        lambda x: x if pd.notna(x) and x != '' and str(x).startswith('http') else 'No URL available'
                    )

                    display_df['Sentiment'] = display_df['compound'].apply(
                        lambda x: '📈 Positive' if x > 0.1 else ('📉 Negative' if x < -0.1 else '⚖️ Neutral')
                    )
                    display_df['Extraction'] = display_df['extraction_success'].apply(
                        lambda x: '✅ Full Content' if x else '📰 Title Only'
                    )
                    display_df['Score'] = display_df['compound'].round(3)

                    # Rename columns for display (omit Scraper Used)
                    display_columns = {
                        'ticker': 'Symbol',
                        'title_short': 'Article Title',
                        'url_clean': 'News URL',
                        'Sentiment': 'Trend',
                        'Score': 'Score',
                        'Extraction': 'Content'
                    }

                    display_df_final = display_df[['ticker', 'title_short', 'url_clean', 'Sentiment', 'Score', 'Extraction']].copy()
                    display_df_final.columns = list(display_columns.values())
                    
                    # Mostrar información sobre la tabla
                    st.info(f"**{len(display_df_final)} articles** analyzed in total.")

                    # Convertir la columna 'News URL' en enlaces HTML para abrir en nueva pestaña.
                    html_df = display_df_final.copy()
                    def _to_link(u):
                        try:
                            if isinstance(u, str) and u.startswith('http'):
                                return f'<a href="{u}" target="_blank" rel="noopener noreferrer">Open</a>'
                        except Exception:
                            pass
                        return u

                    if 'News URL' in html_df.columns:
                        html_df['News URL'] = html_df['News URL'].apply(_to_link)

                    # Styling for the HTML table to make it responsive and clickable
                    st.markdown("""
                    <style>
                    .styled-table { width:100%; border-collapse: collapse; font-family: inherit; }
                    .styled-table th, .styled-table td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); text-align: left; }
                    .styled-table th { background: rgba(255,255,255,0.02); }
                    .styled-table a { color: #00d4ff; text-decoration: none; }
                    .styled-table a:hover { text-decoration: underline; }
                    .table-wrapper { width:100%; overflow:auto; max-height:600px; }
                    </style>
                    """, unsafe_allow_html=True)

                    # Renderizar tabla como HTML para evitar comportamiento de selección/celda vacío
                    html_table = html_df.to_html(index=False, escape=False, classes='styled-table')
                    st.markdown(f'<div class="table-wrapper">{html_table}</div>', unsafe_allow_html=True)
                    
                    # Cerrar última sección
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("⚠️ Please enter at least one valid stock symbol.")

def main():
    """
    Main application with modern landing page
    """
    # Page config set at module import
    
    # Custom CSS for modern landing page
    st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background-color: #000000 !important;
    }
    
    .main .block-container {
        background-color: #000000 !important;
        padding-top: 0rem;
    }
    
    .main-header {
        text-align: center;
        padding: 60px 0 40px 0;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 2px solid #10a37f;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #10a37f;
        margin: 0 0 2rem 0;
        font-weight: 500;
    }
    
    .hero-description {
        font-size: 1.1rem;
        color: #ccc;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(16, 163, 127, 0.2);
        border-color: #10a37f;
    }
    
    .clickable-card {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 10;
        background: transparent;
        border: none;
        cursor: pointer;
    }
    
    /* Link wrapper for clickable cards */
    a.card-link {
        text-decoration: none;
        color: inherit;
        display: block;
    }
    a.card-link:focus { outline: none; }
    
    /* Make cards clickable with hover effects */
    .clickable-card {
        transition: all 0.3s ease;
        position: relative;
    }
    
    .clickable-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(16, 163, 127, 0.3);
        border-color: #10a37f;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 1rem 0;
    }
    
    .feature-description {
        color: #ccc;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #10a37f 0%, #0d8a66 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(16, 163, 127, 0.4);
    }
    
    .stats-container {
        background: #0a0a0a;
        padding: 40px 0;
        margin: 3rem -1rem 2rem -1rem;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
    }
    
    .stat-item {
        text-align: center;
        color: white;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10a37f;
        display: block;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #ccc;
        margin-top: 0.5rem;
    }
    
    .navigation-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .nav-pill {
        background: #1a1a1a;
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        border: 1px solid #333;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .nav-pill:hover, .nav-pill.active {
        background: #10a37f;
        border-color: #10a37f;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "landing"
    
    # Sync state with URL query params
    try:
    params = st.query_params
        if "page" in params:
            page_val = params["page"][0] if isinstance(params["page"], list) else params["page"]
            if page_val in ("landing", "sentiment", "investment", "kpis_chat"):
                st.session_state.current_page = page_val
    except Exception:
        pass
    
    # Landing Page
    if st.session_state.current_page == "landing":
        show_landing_page()
    elif st.session_state.current_page == "sentiment":
        show_sentiment_analysis()
    elif st.session_state.current_page == "investment":
        show_investment_chatbot()
    elif st.session_state.current_page == "kpis_chat":
        show_finviz_dashboard_chat()

def show_landing_page():
    """
    Display the main landing page
    """
    # Hero Section with your logo
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem;">
          <img src="https://i.postimg.cc/PrPKRBTq/Foto-COMERCIANDOLA.png" 
              style="height: 140px; margin-right: 20px; border-radius: 10px;" 
                 alt="ComercianDola Logo">
            <div>
                <h1 class="hero-title" style="margin: 0;">Stockfeels.com</h1>
                <p class="hero-subtitle" style="margin: 0;">Professional Financial Analysis Tools</p>
            </div>
        </div>
        <p class="hero-description">
            Advanced AI-powered stock analysis platform combining sentiment analysis, 
            investment strategies, and market intelligence for smarter trading decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic session banner: show user email and credits when logged in, otherwise sign-in CTA
    try:
        user_obj = st.session_state.get('user') if 'user' in st.session_state else None
        if user_obj:
            user_email = user_obj.get('email') if isinstance(user_obj, dict) else getattr(user_obj, 'email', '')
            user_id = user_obj.get('id') if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
            try:
                credits_val = get_credits(user_id) if user_id else None
            except Exception:
                credits_val = None
            st.markdown(f"""
            <div style='max-width:1200px;margin:0.5rem auto;padding:12px 20px;border-radius:10px;background:#072227;border:1px solid rgba(255,255,255,0.03);display:flex;justify-content:space-between;align-items:center;'>
                <div style='color:#fff;font-weight:700;'>Session started with <span style='color:#10a37f;margin-left:8px;'>{user_email}</span></div>
                <div style='display:flex;gap:12px;align-items:center;'>
                    <div id='credits-bubble-home' style='background:linear-gradient(90deg,#0ea37f,#076b57);padding:6px 10px;border-radius:999px;color:#fff;font-weight:700;'>Credits: {credits_val if credits_val is not None else 'N/A'}</div>
                    <a href='?action=logout' style='text-decoration:none;'><div style='background:#111;border:1px solid rgba(255,255,255,0.04);padding:8px 12px;border-radius:10px;color:#fff;font-weight:700;cursor:pointer;'>Logout</div></a>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # If user just logged out, optionally redirect to sentiment page where login form lives
            show_login = st.session_state.get('show_login_on_landing', False)
            if show_login:
                js = "<script>window.location.search = '?page=sentiment';</script>"
                st.markdown(js, unsafe_allow_html=True)
            else:
                # Render a Streamlit-native banner so we can show a real button that toggles an inline login form
                banner_cols = st.columns([3, 1])
                with banner_cols[0]:
                    st.markdown("""
                    <div style='color:#fff;font-weight:700;padding:6px 0;'>No account is currently signed in</div>
                    """, unsafe_allow_html=True)
                with banner_cols[1]:
                    if st.button('Sign in / Register', key='landing_signin_btn'):
                        st.session_state['show_landing_login'] = True
                        # keep user on landing and show the inline login/register form
                        st.experimental_rerun()

                    # If the user asked to show the login form on the landing page, render it inline here
                    if st.session_state.get('show_landing_login'):
                        st.markdown("""
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                            <div style="width:44px;height:44px;border-radius:50%;background:#10a37f;display:flex;align-items:center;justify-content:center;font-size:20px;">🔐</div>
                            <div style="color:#ffffff;font-weight:700;font-size:20px;">Login or Register</div>
                        </div>
                        """, unsafe_allow_html=True)
                        # Email input with custom white label
                        st.markdown("""
                        <div style='margin-bottom:6px;'><label style='color:#ffffff;font-weight:600;'>Email</label></div>
                        """, unsafe_allow_html=True)
                        email = st.text_input('', key='auth_email', placeholder='you@domain.com', label_visibility='collapsed')
                        # Password input with custom white label
                        st.markdown("""
                        <div style='margin-top:6px;margin-bottom:6px;'><label style='color:#ffffff;font-weight:600;'>Password</label></div>
                        """, unsafe_allow_html=True)
                        password = st.text_input('', type='password', key='auth_password', placeholder='Your password', label_visibility='collapsed')
                        # Remember me rendered as checkbox + white label
                        cb_col, lbl_col = st.columns([1, 20])
                        with cb_col:
                            remember_val = st.checkbox('', value=False, key='remember_me')
                        with lbl_col:
                            st.markdown("""
                            <div style='color:#ffffff; margin-top:8px;'>Remember me on this device</div>
                            """, unsafe_allow_html=True)
                        remember_me = remember_val
                        coll1, coll2 = st.columns(2)
                        with coll1:
                            if st.button('Login', key='landing_login_btn'):
                                ok, resp = sign_in(email, password)
                                if ok:
                                            user_obj = None
                                            if isinstance(resp, dict):
                                                user_obj = resp.get('user') or resp.get('data', {}).get('user') or resp.get('data', {}).get('session', {}).get('user') or resp
                                            else:
                                                user_obj = getattr(resp, 'user', None) or getattr(resp, 'data', None) or resp

                                            st.session_state['user'] = user_obj
                                            try:
                                                # always persist locally
                                                save_local_session(user_obj)
                                                st.session_state['remember_me'] = True
                                            except Exception:
                                                pass
                                            # ensure user record exists
                                            try:
                                                uid = user_obj.get('id') if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
                                                if uid:
                                                    create_user_record_if_missing(uid, user_obj.get('email') if isinstance(user_obj, dict) else getattr(user_obj, 'email', ''))
                                            except Exception:
                                                pass
                                            # hide the inline login form after successful login
                                            st.session_state['show_landing_login'] = False
                                            st.experimental_rerun()
                        with coll2:
                            if st.button('Register', key='landing_register_btn'):
                                ok, resp = sign_up(email, password)
                                if ok:
                                    # Show message instructing user to confirm email (match sentiment section)
                                    st.success('Registration received')
                                    st.info('A confirmation email has been sent by Supabase. Please confirm your email to continue.')
                                    st.warning('You must confirm your email to receive the 300 credits.')
    except Exception:
        pass

    # Features Section - Clickable Cards
    st.markdown("""
    <div style="max-width: 1200px; margin: 3rem auto; padding: 0 2rem;">
        <h2 style="text-align: center; color: white; font-size: 2.5rem; margin-bottom: 3rem;">
            Choose Your Analysis Tool
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        # Clickable card via URL param
        st.markdown("""
        <a class="card-link" href="?page=sentiment">
            <div class="feature-card clickable-card" style="cursor: pointer;">
                <span class="feature-icon">📈</span>
                <h3 class="feature-title">Sentiment Analysis</h3>
                <p class="feature-description">
                    Analyze market sentiment from news articles using advanced AI. 
                    Get real-time sentiment scores, trends, and intelligent summaries 
                    for any stock symbol to make informed trading decisions.
                </p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
    <a class="card-link" href="?page=kpis_chat">
            <div class="feature-card clickable-card" style="cursor: pointer;">
                <span class="feature-icon">🧭</span>
                <h3 class="feature-title">KPI Dashboard & Chat</h3>
                <p class="feature-description">
                    A beautiful KPI dashboard that presents the most important indicators at a glance, followed by an AI chat below to discuss the data and answer your questions.
                </p>
            </div>
        </a>
        """, unsafe_allow_html=True)

    with c3:
        # Clickable card via URL param
        st.markdown("""
    <a class="card-link" href="?page=investment">
            <div class="feature-card clickable-card" style="cursor: pointer;">
                <span class="feature-icon">🎯</span>
                <h3 class="feature-title">Investment Strategy Assistant</h3>
                <p class="feature-description">
                    Get personalized Finviz screening strategies and put options guidance. 
                    AI-powered chatbot helps you find the perfect stocks for your 
                    investment style and risk tolerance.
                </p>
            </div>
        </a>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <a class="card-link" href="#" style="text-decoration:none;">
            <div class="feature-card clickable-card" style="cursor: pointer; border: 1px dashed rgba(255,255,255,0.06);">
                <span class="feature-icon">🚀</span>
                <h3 class="feature-title">Advanced Research & Alerts</h3>
                <p class="feature-description">
                    <strong>Coming soon</strong>: Insider activity — see what the market's big players are doing.
                </p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    # API Logos Carousel Section
    # Access & Credits informational section
    st.markdown("""
    <div style="max-width: 1200px; margin: 2rem auto; text-align: center;">
        <div class="glass-card" style="padding:1.25rem; background: rgba(255,255,255,0.03); border:1px solid #2c2c2c;">
            <h3 style="margin-bottom:0.25rem;">Access &amp; Credits</h3>
            <p style="margin-top:0.5rem; color: #ddd;">
                To run analyses you must be signed in. Each new account receives <strong>300 credits</strong>.
                Starting a sentiment analysis costs <strong>50 credits</strong>, while a conversation with the <strong>Investment</strong> assistant costs <strong>100 credits</strong>.
                Credits are deducted automatically when you press <em>Start Analysis</em> or when you submit an investment request.
                    <br>
                    The <strong>Investment</strong> page hosts an AI-powered Investment Assistant (Finviz screening, strategy guidance and options help). It requires you to be signed in and may consume credits for advanced queries or extended sessions.
                    <br>
                    Pay <strong>$12</strong> to upgrade to unlimited Sentiment, Investment and KPI Dashboard &amp; Chat access.
            </p>
            <p style="margin-top:0.25rem; color: #fff; font-size: 0.95rem;">
                Create an account or log in to track your credits and access the full platform.
            </p>
            <div style="margin-top:0.75rem;">
                <a href="https://www.paypal.com/ncp/payment/HDFJJ2VJHZBD4" target="_blank" style="text-decoration:none;">
                    <div style="display:inline-block;background:linear-gradient(90deg,#ff8a00,#ff3d00);color:#111;padding:10px 14px;border-radius:10px;font-weight:800;box-shadow:0 8px 20px rgba(255,60,0,0.18);">I want unlimited credits for $12</div>
                </a>
            </div>
        </div>
    </div>

    """, unsafe_allow_html=True)

    # API Logos Animated Carousel
    st.markdown("""
    <style>
    .api-carousel-viewport { overflow: hidden; }
    .api-carousel-track {
      display: flex;
      gap: 3rem;
      align-items: center;
      animation: scroll-left 18s linear infinite;
    }
    .api-carousel-track:hover { animation-play-state: paused; }
    .api-carousel-track img { height: 48px; filter: brightness(0.85); }

    @keyframes scroll-left {
      0% { transform: translateX(0); }
      50% { transform: translateX(-40%); }
      100% { transform: translateX(0); }
    }
    </style>

    <div style="margin: 4rem 0; padding: 3rem 0; background: linear-gradient(to bottom, rgba(10,10,10,0.5), rgba(0,0,0,1)); border-top: 1px solid #333; border-bottom: 1px solid #333;">
        <div style="max-width: 1200px; margin: 0 auto; text-align: center;">
            <h3 style="color: white; font-size: 2rem; margin-bottom: 2rem; font-weight: 600;">
                APIs that Power Our Services
            </h3>
            <div class="api-carousel-viewport" style="width:100%;">
                <div class="api-carousel-track" style="width:200%;">
                    <!-- Duplicate logos for seamless looping -->
                    <img src="https://i.postimg.cc/CKqLzGHT/1681142503openai-icon-png-1.png" alt="OpenAI">
                    <img src="https://coursevania.com/wp-content/uploads/2023/02/5021180_dbc8_2.jpg" alt="Coursevania">
                    <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="Google Cloud">
                    <img src="https://i.ibb.co/DrqZ7Sc/logo-6813a1b5.png" alt="Vapi">
                    <img src="https://i.ibb.co/M6XR8VZ/img-swapcard.png" alt="Deepgram">
                    <img src="https://i.ibb.co/VpqN1Kc/Bolt.jpg" alt="Bolt.new">
                    <img src="https://i.ibb.co/92JrX6Z/logo-make.png" alt="Make.com">
                    <img src="https://i.ibb.co/zJmdcF0/hostinger-logo-freelogovectors-net.png" alt="Hostinger">
                    <!-- duplicated set -->
                    <img src="https://i.postimg.cc/CKqLzGHT/1681142503openai-icon-png-1.png" alt="OpenAI">
                    <img src="https://coursevania.com/wp-content/uploads/2023/02/5021180_dbc8_2.jpg" alt="Coursevania">
                    <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="Google Cloud">
                    <img src="https://i.ibb.co/DrqZ7Sc/logo-6813a1b5.png" alt="Vapi">
                    <img src="https://i.ibb.co/M6XR8VZ/img-swapcard.png" alt="Deepgram">
                    <img src="https://i.ibb.co/VpqN1Kc/Bolt.jpg" alt="Bolt.new">
                    <img src="https://i.ibb.co/92JrX6Z/logo-make.png" alt="Make.com">
                    <img src="https://i.ibb.co/zJmdcF0/hostinger-logo-freelogovectors-net.png" alt="Hostinger">
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional Features Section
    st.markdown("""
    <div style="max-width: 1200px; margin: 4rem auto 2rem auto; padding: 0 2rem;">
        <h2 style="text-align: center; color: white; font-size: 2rem; margin-bottom: 2rem;">
            Why Choose Stockfeels.com?
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <span style="font-size: 2.5rem; color: #10a37f;">🤖</span>
            <h4 style="color: white; margin: 1rem 0 0.5rem 0;">AI-Powered</h4>
            <p style="color: #ccc; font-size: 0.9rem;">
                Advanced GPT-5 integration for intelligent analysis and recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <span style="font-size: 2.5rem; color: #10a37f;">⚡</span>
            <h4 style="color: white; margin: 1rem 0 0.5rem 0;">Real-Time</h4>
            <p style="color: #ccc; font-size: 0.9rem;">
                Live market data and instant sentiment analysis for quick decisions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <span style="font-size: 2.5rem; color: #10a37f;">🎯</span>
            <h4 style="color: white; margin: 1rem 0 0.5rem 0;">Personalized</h4>
            <p style="color: #ccc; font-size: 0.9rem;">
                Tailored strategies based on your investment style and risk tolerance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding: 2rem; border-top: 1px solid #404040; color: #666;">
        <p style="margin: 0; font-size: 0.9rem;">
            Built with ❤️ for smarter investing • Powered by AI • Real-time Market Data
        </p>
        <p style="margin: 6px 0 0 0; font-size: 0.75rem; color: #999; max-width:800px; margin-left:auto; margin-right:auto;">
            Disclaimer: Content is for informational purposes only and does not constitute financial advice. Market data may be delayed or inaccurate. Users should verify information independently and consult a licensed professional for investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_sentiment_analysis():
    """
    Display sentiment analysis with back navigation
    """
    # Back button and title
    col1, col2 = st.columns([1, 8])
    
    with col1:
        if st.button("← Back", type="secondary"):
            st.session_state.current_page = "landing"
            try:
                st.experimental_set_query_params(page="landing")
            except Exception:
                pass
            st.experimental_rerun()
    
    with col2:
        # Title is handled by show_page() - no duplicate needed here
        pass
    
    st.markdown("---")
    
    # Call the original sentiment analysis function
    show_page()

if __name__ == "__main__":
    main()