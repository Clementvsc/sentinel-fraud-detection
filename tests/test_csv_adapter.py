from sentinel.datasets.csv_adapter import load_csv_events

_SPARKOV = """trans_date_trans_time,cc_num,merchant,category,amt,city,lat,long,is_fraud
2025-06-01 12:00:00,1234567890,fraud_Store A,grocery_pos,42.10,NYC,40.71,-74.01,0
2025-06-01 12:05:00,1234567890,fraud_Store B,shopping_net,9.99,NYC,40.71,-74.01,1
"""

_ULB = "Time," + ",".join(f"V{i}" for i in range(1, 29)) + ",Amount,Class\n" + \
       "0," + ",".join("0.1" for _ in range(28)) + ",10.0,0\n" + \
       "5," + ",".join("0.2" for _ in range(28)) + ",20.0,1\n"

_UPI = """timestamp,amount,currency,upi_app,bank,device_fingerprint,status,is_suspicious
2025-12-23T09:57:17.900016,1815.98,INR,Amazon Pay,Axis,device_abc,success,False
2025-12-23T09:57:41.163066,48.52,INR,GPay,SBI,device_xyz,success,True
2025-12-23T09:58:00.000000,999.00,INR,PhonePe,HDFC,device_abc,failed,False
"""


def test_sparkov_mapping(tmp_path):
    p = tmp_path / "sp.csv"
    p.write_text(_SPARKOV)
    ev = load_csv_events(str(p), "sparkov")
    assert [e["label"] for e in ev] == [0, 1]
    assert ev[0]["mcc"] == "grocery" and ev[0]["cust_id"] == "cc_1234567890"
    assert ev[1]["channel"] == "online"
    assert ev[0]["ts"] < ev[1]["ts"]


def test_ulb_raw_features(tmp_path):
    p = tmp_path / "ulb.csv"
    p.write_text(_ULB)
    ev = load_csv_events(str(p), "ulb")
    assert ev[0]["raw_features"]["V1"] == 0.1
    assert ev[1]["raw_features"]["Amount"] == 20.0
    assert [e["label"] for e in ev] == [0, 1]


def test_upi_mapping(tmp_path):
    p = tmp_path / "upi.csv"
    p.write_text(_UPI)
    ev = load_csv_events(str(p), "upi")
    # the failed transaction is dropped — a declined attempt never redefines
    # "normal" behaviour, same principle ProfileState.update() applies
    assert len(ev) == 2
    assert [e["label"] for e in ev] == [0, 1]
    assert ev[0]["cust_id"] == "upi_device_abc"
    assert ev[0]["country"] == "IN" and ev[0]["channel"] == "transfer"
    assert ev[0]["merchant_id"] == "Axis_Amazon Pay"
    assert ev[0]["amount"] == 1815.98
    assert ev[0]["ts"] < ev[1]["ts"]


def test_unknown_schema(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    try:
        load_csv_events(str(p), "nope")
        assert False
    except ValueError:
        pass
