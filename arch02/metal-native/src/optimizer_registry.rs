//! Stable optimizer names shared by the CLI, checkpoints, FFI, and research manifests.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    MuonNs5Adamw,
    MuonNs3Adamw,
    MuonPolarAdamw,
    NorMuonAdamw,
    MuownAdamw,
    MonaAdamw,
    MiMuonAdamw,
    Adamw,
    Lion,
    CautiousAdamw,
    CautiousLion,
    SgdMomentum,
    Sophia,
    ScheduleFreeAdamw,
    Prodigy,
    SoapAdamw,
}

impl Default for OptimizerKind {
    fn default() -> Self {
        Self::MuonNs5Adamw
    }
}

impl OptimizerKind {
    pub const ALL: [Self; 16] = [
        Self::MuonNs5Adamw,
        Self::MuonNs3Adamw,
        Self::MuonPolarAdamw,
        Self::NorMuonAdamw,
        Self::MuownAdamw,
        Self::MonaAdamw,
        Self::MiMuonAdamw,
        Self::Adamw,
        Self::Lion,
        Self::CautiousAdamw,
        Self::CautiousLion,
        Self::SgdMomentum,
        Self::Sophia,
        Self::ScheduleFreeAdamw,
        Self::Prodigy,
        Self::SoapAdamw,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::MuonNs5Adamw => "muon_ns5_adamw",
            Self::MuonNs3Adamw => "muon_ns3_adamw",
            Self::MuonPolarAdamw => "muon_polar_adamw",
            Self::NorMuonAdamw => "normuon_adamw",
            Self::MuownAdamw => "muown_adamw",
            Self::MonaAdamw => "mona_adamw",
            Self::MiMuonAdamw => "mimuon_adamw",
            Self::Adamw => "adamw",
            Self::Lion => "lion",
            Self::CautiousAdamw => "cautious_adamw",
            Self::CautiousLion => "cautious_lion",
            Self::SgdMomentum => "sgd_momentum",
            Self::Sophia => "sophia",
            Self::ScheduleFreeAdamw => "schedule_free_adamw",
            Self::Prodigy => "prodigy",
            Self::SoapAdamw => "soap_adamw",
        }
    }

    /// Native hot-path status.  Paper-only candidates stay explicit instead of
    /// silently falling back to the Muon anchor.
    pub fn native_ready(self) -> bool {
        matches!(
            self,
            Self::MuonNs5Adamw
                | Self::MuonNs3Adamw
                | Self::MuonPolarAdamw
                | Self::NorMuonAdamw
                | Self::MonaAdamw
                | Self::MuownAdamw
                | Self::Adamw
                | Self::Lion
                | Self::CautiousAdamw
                | Self::CautiousLion
                | Self::SgdMomentum
                | Self::Sophia
                | Self::ScheduleFreeAdamw
                | Self::Prodigy
        )
    }

    pub fn uses_muon_clip(self) -> bool {
        matches!(
            self,
            Self::MuonNs5Adamw
                | Self::MuonNs3Adamw
                | Self::MuonPolarAdamw
                | Self::NorMuonAdamw
                | Self::MuownAdamw
                | Self::MonaAdamw
                | Self::MiMuonAdamw
        )
    }
}

impl fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OptimizerKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.to_ascii_lowercase().replace('-', "_");
        let kind = match normalized.as_str() {
            "muon" | "muon_ns5" | "muon_ns5_adamw" => Self::MuonNs5Adamw,
            "neomuon" | "muon_ns3" | "muon_ns3_adamw" => Self::MuonNs3Adamw,
            "polar" | "muon_polar" | "muon_polar_adamw" => Self::MuonPolarAdamw,
            "normuon" | "normuon_adamw" => Self::NorMuonAdamw,
            "muown" | "muown_adamw" => Self::MuownAdamw,
            "mona" | "mona_adamw" => Self::MonaAdamw,
            "mimuon" | "mimuon_adamw" => Self::MiMuonAdamw,
            "adam" | "adamw" => Self::Adamw,
            "lion" => Self::Lion,
            "c_adamw" | "cautious_adamw" => Self::CautiousAdamw,
            "c_lion" | "cautious_lion" => Self::CautiousLion,
            "sgd" | "momentum" | "sgd_momentum" => Self::SgdMomentum,
            "sophia" => Self::Sophia,
            "schedulefree" | "schedule_free" | "schedule_free_adamw" => Self::ScheduleFreeAdamw,
            "prodigy" => Self::Prodigy,
            "soap" | "soap_adamw" => Self::SoapAdamw,
            _ => {
                return Err(format!(
                    "unknown optimizer '{value}'; valid names: {}",
                    Self::ALL
                        .iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                ))
            }
        };
        Ok(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_names_are_stable() {
        assert_eq!("neomuon".parse(), Ok(OptimizerKind::MuonNs3Adamw));
        assert_eq!(
            "schedule-free".parse(),
            Ok(OptimizerKind::ScheduleFreeAdamw)
        );
        for kind in OptimizerKind::ALL {
            assert_eq!(kind.as_str().parse::<OptimizerKind>().unwrap(), kind);
        }
    }
}
