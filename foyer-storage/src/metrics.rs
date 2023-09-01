//  Copyright 2023 MrCroxx
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

use std::sync::LazyLock;

use prometheus::{
    register_histogram_vec, register_int_counter_vec, register_int_gauge_vec, Histogram,
    HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
};

/// Multiple foyer instance will share the same global metrics with different label `foyer` name.
pub static METRICS: LazyLock<GlobalMetrics> = LazyLock::new(GlobalMetrics::default);

#[derive(Debug)]
pub struct GlobalMetrics {
    op_duration: HistogramVec,
    op_bytes: IntCounterVec,
    total_bytes: IntGaugeVec,
}

impl Default for GlobalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalMetrics {
    pub fn new() -> Self {
        let op_duration = register_histogram_vec!(
            "foyer_storage_op_duration",
            "foyer storage op duration",
            &["foyer", "op", "extra"],
            vec![0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0],
        )
        .unwrap();

        let op_bytes = register_int_counter_vec!(
            "foyer_storage_op_bytes",
            "foyer storage op bytes",
            &["foyer", "op", "extra"]
        )
        .unwrap();

        let total_bytes =
            register_int_gauge_vec!("total_bytes", "total bytes", &["foyer"]).unwrap();

        Self {
            op_duration,
            op_bytes,
            total_bytes,
        }
    }

    pub fn foyer(&self, name: &str) -> Metrics {
        Metrics::new(self, name)
    }
}

#[derive(Debug)]
pub struct Metrics {
    pub op_duration_insert_inserted: Histogram,
    pub op_duration_insert_filtered: Histogram,
    pub op_duration_insert_dropped: Histogram,
    pub op_duration_lookup_hit: Histogram,
    pub op_duration_lookup_miss: Histogram,
    pub op_duration_remove: Histogram,
    pub op_duration_flush: Histogram,
    pub op_duration_reclaim: Histogram,

    pub op_bytes_insert: IntCounter,
    pub op_bytes_lookup: IntCounter,
    pub op_bytes_flush: IntCounter,
    pub op_bytes_reclaim: IntCounter,
    pub op_bytes_reinsert: IntCounter,

    pub total_bytes: IntGauge,
}

impl Metrics {
    pub fn new(global: &GlobalMetrics, foyer: &str) -> Self {
        let op_duration_insert_inserted = global
            .op_duration
            .with_label_values(&[foyer, "insert", "inserted"]);
        let op_duration_insert_filtered = global
            .op_duration
            .with_label_values(&[foyer, "insert", "filtered"]);
        let op_duration_insert_dropped = global
            .op_duration
            .with_label_values(&[foyer, "insert", "dropped"]);
        let op_duration_lookup_hit = global
            .op_duration
            .with_label_values(&[foyer, "lookup", "hit"]);
        let op_duration_lookup_miss = global
            .op_duration
            .with_label_values(&[foyer, "lookup", "miss"]);
        let op_duration_remove = global.op_duration.with_label_values(&[foyer, "remove", ""]);
        let op_duration_flush = global.op_duration.with_label_values(&[foyer, "flush", ""]);
        let op_duration_reclaim = global
            .op_duration
            .with_label_values(&[foyer, "reclaim", ""]);

        let op_bytes_insert = global.op_bytes.with_label_values(&[foyer, "insert", ""]);
        let op_bytes_lookup = global.op_bytes.with_label_values(&[foyer, "lookup", ""]);
        let op_bytes_flush = global.op_bytes.with_label_values(&[foyer, "flush", ""]);
        let op_bytes_reclaim = global.op_bytes.with_label_values(&[foyer, "reclaim", ""]);
        let op_bytes_reinsert = global.op_bytes.with_label_values(&[foyer, "reinsert", ""]);

        let total_bytes = global.total_bytes.with_label_values(&[foyer]);

        Self {
            op_duration_insert_inserted,
            op_duration_insert_filtered,
            op_duration_insert_dropped,
            op_duration_lookup_hit,
            op_duration_lookup_miss,
            op_duration_remove,
            op_duration_flush,
            op_duration_reclaim,
            op_bytes_insert,
            op_bytes_lookup,
            op_bytes_flush,
            op_bytes_reclaim,
            op_bytes_reinsert,
            total_bytes,
        }
    }
}
