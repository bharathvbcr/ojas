//! The bytes Python hashes, reproduced: `repr`, `float.hex`, canonical `json.dumps`,
//! `ConsumedPrefix`, `LossLog.digest` and `datetime.isoformat`, against what
//! `tools/qd_train_oracle_trainer.py` dumped from Python itself.

mod common;

use std::time::{Duration, UNIX_EPOCH};

use ojas_io::json::JsonValue as Value;
use ojas_train::lappi::pyjson::{
    CANONICAL, CANONICAL_ASCII, dumps, float, float_hex, float_repr, obj,
};
use ojas_train::lappi::run_control::ConsumedPrefix;
use ojas_train::lappi::run_control::{LossLog, LossPoint};

/// `datetime.now(UTC).isoformat()` for `t`: microseconds when non-zero, `+00:00`. The same
/// civil-from-days arithmetic as Lappi's `ledger::isoformat_utc`, which does not move with
/// this crate; only this oracle needs it.
fn isoformat_utc(t: std::time::SystemTime) -> Result<String, String> {
    let d = t
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("a time before 1970: {e}"))?;
    let secs = i64::try_from(d.as_secs()).map_err(|e| e.to_string())?;
    let micros = d.subsec_micros();
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let frac = if micros == 0 {
        String::new()
    } else {
        format!(".{micros:06}")
    };
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}{frac}+00:00"
    ))
}

#[test]
fn repr_and_hex_match_python_on_every_dumped_float() {
    let o = common::trainer_oracle();
    let cases = o["floats"].as_array().unwrap();
    assert!(cases.len() > 600, "{} floats", cases.len());
    for c in cases {
        let x = f64::from_bits(c["bits"].as_u64().unwrap());
        assert_eq!(
            float_repr(x),
            c["repr"].as_str().unwrap(),
            "repr of bits {:#x}",
            x.to_bits()
        );
        assert_eq!(
            float_hex(x).unwrap(),
            c["hex"].as_str().unwrap(),
            "hex of bits {:#x}",
            x.to_bits()
        );
    }
}

/// Rebuild a fixture value (`{"$f": hex}`, `{"$int": n}`, `{"$obj": [[k, v], ...]}`).
fn rebuild(v: &Value) -> Value {
    match v {
        Value::Null | Value::Bool(_) | Value::String(_) => v.clone(),
        Value::Array(xs) => Value::Array(xs.iter().map(rebuild).collect()),
        Value::Object(_) => {
            if let Some(h) = v.field_opt("$f").unwrap() {
                float(common::fhex(h)).unwrap()
            } else if let Some(i) = v.field_opt("$int").unwrap() {
                Value::from(i.as_i64().unwrap())
            } else {
                let pairs = v["$obj"].as_array().unwrap().iter().map(|kv| {
                    let kv = kv.as_array().unwrap();
                    (kv[0].as_str().unwrap().to_string(), rebuild(&kv[1]))
                });
                obj(pairs).unwrap()
            }
        }
        Value::Number(_) => panic!("the fixture tags every number"),
    }
}

#[test]
fn canonical_dumps_is_pythons_byte_for_byte() {
    let o = common::trainer_oracle();
    for c in o["json"].as_array().unwrap() {
        let v = rebuild(&c["value"]);
        assert_eq!(
            dumps(&v, CANONICAL_ASCII).unwrap(),
            c["ascii"].as_str().unwrap()
        );
        assert_eq!(dumps(&v, CANONICAL).unwrap(), c["utf8"].as_str().unwrap());
    }
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn consumed_prefix_digests_are_pythons() {
    let o = common::trainer_oracle();
    for c in o["consumed"].as_array().unwrap() {
        let mut p = ConsumedPrefix::new();
        for fold in c["folds"].as_array().unwrap() {
            let parts: Vec<Vec<u8>> = fold
                .as_array()
                .unwrap()
                .iter()
                .map(|h| unhex(h.as_str().unwrap()))
                .collect();
            let refs: Vec<&[u8]> = parts.iter().map(|v| v.as_slice()).collect();
            p.fold(&refs);
        }
        assert_eq!(p.hexdigest(), c["digest"].as_str().unwrap());
    }
}

#[test]
fn the_loss_log_digest_is_pythons() {
    let o = common::trainer_oracle();
    let case = &o["loss_log"];
    let mut log = LossLog::new();
    for p in case["points"].as_array().unwrap() {
        let p = p.as_array().unwrap();
        log.append(LossPoint {
            optimizer_step: p[0].as_u64().unwrap(),
            epoch: p[1].as_u64().unwrap(),
            batch_index: p[2].as_u64().unwrap(),
            loss: common::fhex(&p[3]),
        })
        .unwrap();
    }
    assert_eq!(
        dumps(&log.to_json().unwrap(), CANONICAL_ASCII).unwrap(),
        case["json"].as_str().unwrap()
    );
    assert_eq!(log.digest().unwrap(), case["digest"].as_str().unwrap());
}

#[test]
fn isoformat_is_pythons() {
    let o = common::trainer_oracle();
    for c in o["isoformat"].as_array().unwrap() {
        let t = UNIX_EPOCH
            + Duration::from_secs(c["secs"].as_u64().unwrap())
            + Duration::from_micros(c["micros"].as_u64().unwrap());
        assert_eq!(isoformat_utc(t).unwrap(), c["iso"].as_str().unwrap());
    }
}
