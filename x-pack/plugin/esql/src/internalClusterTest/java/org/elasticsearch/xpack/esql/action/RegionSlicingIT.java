/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.esql.action;

import org.elasticsearch.action.bulk.BulkRequestBuilder;
import org.elasticsearch.action.support.WriteRequest;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.compute.lucene.query.LuceneSourceOperator;
import org.elasticsearch.compute.operator.DriverProfile;
import org.elasticsearch.xpack.esql.plugin.ComputeService;
import org.elasticsearch.xpack.esql.plugin.QueryPragmas;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;

import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertAcked;
import static org.elasticsearch.xpack.esql.EsqlTestUtils.getValuesList;
import static org.elasticsearch.xpack.esql.action.EsqlQueryRequest.syncEsqlQueryRequest;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;

/**
 * Runs queries with range filters on the leading fields of the index sort and checks that the results do not depend on the
 * {@link QueryPragmas#REGION_SLICING} pragma. The index has several segments that all contain a run of every counter value.
 */
public class RegionSlicingIT extends AbstractEsqlIntegTestCase {
    private static final String INDEX = "region_slicing";
    private static final long DAY_MILLIS = 24L * 60 * 60 * 1000;
    private static final long FIRST_DAY = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli();

    private record Row(int counter, long day, long value) {}

    private final List<Row> rows = new ArrayList<>();

    private void createIndex() {
        assertAcked(
            prepareCreate(INDEX).setSettings(
                Settings.builder()
                    .put("index.number_of_shards", 1)
                    .put("index.number_of_replicas", 0)
                    .putList("index.sort.field", "counter", "day")
                    .putList("index.sort.order", randomFrom("asc", "desc"), randomFrom("asc", "desc"))
            ).setMapping("counter", "type=long", "day", "type=date", "v", "type=long")
        );
        int batches = between(3, 6);
        for (int batch = 0; batch < batches; batch++) {
            BulkRequestBuilder bulk = client().prepareBulk();
            for (int i = 0; i < 1_500; i++) {
                Row row = new Row(randomIntBetween(0, 29), FIRST_DAY + randomIntBetween(0, 4) * DAY_MILLIS, randomIntBetween(0, 9));
                rows.add(row);
                bulk.add(prepareIndex(INDEX).setSource("counter", row.counter, "day", row.day, "v", row.value));
            }
            // A refresh per batch so that the index has several segments.
            bulk.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE).get();
        }
    }

    private QueryPragmas pragmas(boolean regionSlicing) {
        return new QueryPragmas(
            Settings.builder()
                .put(QueryPragmas.REGION_SLICING.getKey(), regionSlicing)
                .put(QueryPragmas.DATA_PARTITIONING.getKey(), "doc")
                .put(QueryPragmas.TASK_CONCURRENCY.getKey(), between(1, 6))
                .build()
        );
    }

    private List<List<Object>> run(String query, boolean regionSlicing) {
        EsqlQueryRequest request = syncEsqlQueryRequest(query);
        request.pragmas(pragmas(regionSlicing));
        request.acceptedPragmaRisks(true);
        try (EsqlQueryResponse response = run(request)) {
            List<List<Object>> values = new ArrayList<>();
            getValuesList(response).forEach(values::add);
            return values;
        }
    }

    private void assertStats(String where, Predicate<Row> expected) {
        List<Row> matching = rows.stream().filter(expected).toList();
        long sum = matching.stream().mapToLong(Row::value).sum();
        List<List<Object>> expectedValues = matching.isEmpty()
            ? List.of(Arrays.asList(0L, null))
            : List.of(List.<Object>of((long) matching.size(), sum));
        for (boolean regionSlicing : new boolean[] { true, false }) {
            assertThat(
                "region slicing: " + regionSlicing + " where: " + where,
                run("FROM " + INDEX + " | WHERE " + where + " | STATS c = COUNT(*), s = SUM(v)", regionSlicing),
                equalTo(expectedValues)
            );
        }
    }

    public void testFiltersOnTheLeadingSortFields() {
        assumeTrue("pragmas only enabled on snapshot builds", canUseQueryPragmas());
        createIndex();
        int counter = between(0, 29);
        assertStats("counter == " + counter, r -> r.counter == counter);
        assertStats(
            "counter == " + counter + " AND day >= \"2024-01-02T00:00:00Z\" AND day <= \"2024-01-03T00:00:00Z\"",
            r -> r.counter == counter && r.day >= FIRST_DAY + DAY_MILLIS && r.day <= FIRST_DAY + 2 * DAY_MILLIS
        );
        assertStats(
            "counter == " + counter + " AND day > \"2024-01-02T00:00:00Z\" AND v > 3",
            r -> r.counter == counter && r.day > FIRST_DAY + DAY_MILLIS && r.value > 3
        );
        int low = between(0, 25);
        int high = low + between(0, 3);
        assertStats("counter >= " + low + " AND counter <= " + high, r -> r.counter >= low && r.counter <= high);
        assertStats(
            "counter >= " + low + " AND counter <= " + high + " AND day == \"2024-01-03T00:00:00Z\"",
            r -> r.counter >= low && r.counter <= high && r.day == FIRST_DAY + 2 * DAY_MILLIS
        );
        // nothing matches
        assertStats("counter == 1000", r -> false);
        assertStats("counter == " + counter + " AND day > \"2030-01-01T00:00:00Z\"", r -> false);
        // no filter on the sort fields: the plain partitioning
        assertStats("v == 4", r -> r.value == 4);
    }

    public void testTopNOnRegion() {
        assumeTrue("pragmas only enabled on snapshot builds", canUseQueryPragmas());
        createIndex();
        int counter = between(0, 29);
        int limit = between(1, 20);
        // Ties on both sort keys have the same values, so comparing the values is exact.
        List<Long> expected = rows.stream()
            .filter(r -> r.counter == counter)
            .sorted(Comparator.comparingLong(Row::value).reversed())
            .limit(limit)
            .map(Row::value)
            .toList();
        for (boolean regionSlicing : new boolean[] { true, false }) {
            List<List<Object>> actual = run(
                "FROM " + INDEX + " | WHERE counter == " + counter + " | SORT v DESC | LIMIT " + limit + " | KEEP v",
                regionSlicing
            );
            assertThat("region slicing: " + regionSlicing, actual.stream().map(r -> (Long) r.get(0)).toList(), equalTo(expected));
        }
    }

    /**
     * With region slicing the slices are cut from the doc range of the counter only. Without it the whole segment is cut.
     */
    public void testPragmaChangesTheSlices() {
        assumeTrue("pragmas only enabled on snapshot builds", canUseQueryPragmas());
        assertAcked(
            prepareCreate(INDEX).setSettings(
                Settings.builder()
                    .put("index.number_of_shards", 1)
                    .put("index.number_of_replicas", 0)
                    .putList("index.sort.field", "counter", "day")
                    .putList("index.sort.order", "asc", "asc")
            ).setMapping("counter", "type=long", "day", "type=date", "v", "type=long")
        );
        for (int batch = 0; batch < 20; batch++) {
            BulkRequestBuilder bulk = client().prepareBulk();
            for (int i = 0; i < 5_000; i++) {
                bulk.add(prepareIndex(INDEX).setSource("counter", randomIntBetween(0, 9), "day", FIRST_DAY, "v", 1));
            }
            bulk.get();
        }
        // One big segment: the doc count partitioner then cuts the whole 100k docs, not the 10k of one counter.
        client().admin().indices().prepareForceMerge(INDEX).setMaxNumSegments(1).get();
        client().admin().indices().prepareRefresh(INDEX).get();

        // SUM reads values, so this is a LuceneSourceOperator. A plain COUNT(*) is answered by the count operator, which keeps its own
        // partitioning.
        int onSlices = totalSlices("FROM " + INDEX + " | WHERE counter == 3 | STATS s = SUM(v)", true);
        int offSlices = totalSlices("FROM " + INDEX + " | WHERE counter == 3 | STATS s = SUM(v)", false);
        // About 10k docs for the counter and a task concurrency of 4: 4 slices of about 2.5k docs, above the 1000 docs floor.
        assertThat(onSlices, equalTo(4));
        assertThat(offSlices, not(equalTo(onSlices)));
    }

    private int totalSlices(String query, boolean regionSlicing) {
        EsqlQueryRequest request = syncEsqlQueryRequest(query);
        request.profile(true);
        request.pragmas(
            new QueryPragmas(
                Settings.builder()
                    .put(QueryPragmas.REGION_SLICING.getKey(), regionSlicing)
                    .put(QueryPragmas.DATA_PARTITIONING.getKey(), "doc")
                    .put(QueryPragmas.TASK_CONCURRENCY.getKey(), 4)
                    .build()
            )
        );
        request.acceptedPragmaRisks(true);
        try (EsqlQueryResponse response = run(request)) {
            for (DriverProfile driver : response.profile().drivers()) {
                if (driver.description().equals(ComputeService.DATA_DESCRIPTION)) {
                    return ((LuceneSourceOperator.Status) driver.operators().get(0).status()).totalSlices();
                }
            }
            throw new AssertionError("no data driver in " + response.profile().drivers());
        }
    }
}
