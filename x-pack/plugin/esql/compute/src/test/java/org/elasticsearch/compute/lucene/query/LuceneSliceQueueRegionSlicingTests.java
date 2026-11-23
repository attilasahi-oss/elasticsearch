/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute.lucene.query;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSortSortedNumericDocValuesRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.store.Directory;
import org.elasticsearch.compute.lucene.IndexedByShardIdFromList;
import org.elasticsearch.compute.lucene.PartialLeafReaderContext;
import org.elasticsearch.compute.querydsl.query.QueryWarnings;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;

/**
 * Tests {@link LuceneSliceQueue#create} with and without the region slicing of {@link SortedRegionSlicer}.
 */
public class LuceneSliceQueueRegionSlicingTests extends ESTestCase {
    private static final int DOCS = 120_000;
    private static final int DOCS_PER_VALUE = 6_000;

    /**
     * Builds one sorted segment where doc {@code i} has {@code a == i / DOCS_PER_VALUE}, so {@code a == v} is the doc range
     * {@code [v * DOCS_PER_VALUE, (v + 1) * DOCS_PER_VALUE)}.
     */
    private static void index(Directory directory) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig();
        config.setMergePolicy(NoMergePolicy.INSTANCE);
        config.setIndexSort(new Sort(new SortedNumericSortField("a", SortField.Type.LONG, false)));
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < DOCS; i++) {
                Document doc = new Document();
                doc.add(new SortedNumericDocValuesField("a", i / DOCS_PER_VALUE));
                writer.addDocument(doc);
            }
        }
    }

    private static Query aEquals(long value) {
        // The shape ES builds for a range on a field of an index sorted on that field, wrapped like a pushed down ES|QL filter.
        Query range = new IndexSortSortedNumericDocValuesRangeQuery(
            "a",
            value,
            value,
            SortedNumericDocValuesField.newSlowRangeQuery("a", value, value)
        );
        return new BooleanQuery.Builder().add(range, BooleanClause.Occur.FILTER).build();
    }

    private static List<PartialLeafReaderContext> slices(DirectoryReader reader, Query query, int taskConcurrency, boolean regionSlicing) {
        var shardContext = new LuceneSourceOperatorTests.MockShardContext(reader);
        LuceneSliceQueue queue = LuceneSliceQueue.create(
            new IndexedByShardIdFromList<>(List.of(shardContext)),
            ctx -> List.of(new LuceneSliceQueue.QueryAndTags(query, List.of())),
            DataPartitioning.DOC,
            (ctx, q) -> LuceneSliceQueue.PartitioningStrategy.DOC,
            LuceneOperator.SMALL_INDEX_BOUNDARY,
            taskConcurrency,
            ctx -> ScoreMode.COMPLETE_NO_SCORES,
            LuceneSliceQueue.LeafSplitGuard.NEVER,
            LuceneSliceQueue.MIN_DOCS_PER_SLICE,
            QueryWarnings.NOOP,
            regionSlicing
        );
        List<PartialLeafReaderContext> result = new ArrayList<>();
        LuceneSlice slice = null;
        while ((slice = queue.nextSlice(slice)) != null) {
            assertThat(slice.leaves(), hasSize(1));
            result.add(slice.leaves().get(0));
        }
        assertThat(result.size(), equalTo(queue.totalSlices()));
        return result;
    }

    public void testRegionSlicingCutsOnlyTheRegion() throws IOException {
        try (Directory directory = newDirectory()) {
            index(directory);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                // a == 7 is [42000, 48000): 6000 docs, 4 slices for a concurrency of 4
                List<PartialLeafReaderContext> slices = slices(reader, aEquals(7), 4, true);
                assertThat(slices, hasSize(4));
                int docs = 0;
                for (PartialLeafReaderContext slice : slices) {
                    assertTrue(slice.minDoc() >= 42_000 && slice.maxDoc() <= 48_000);
                    docs += slice.maxDoc() - slice.minDoc();
                }
                assertThat(docs, equalTo(DOCS_PER_VALUE));
            }
        }
    }

    public void testWithoutRegionSlicingTheWholeSegmentIsCut() throws IOException {
        try (Directory directory = newDirectory()) {
            index(directory);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                List<PartialLeafReaderContext> slices = slices(reader, aEquals(7), 4, false);
                int docs = 0;
                for (PartialLeafReaderContext slice : slices) {
                    docs += slice.maxDoc() - slice.minDoc();
                }
                assertThat(docs, equalTo(DOCS));
            }
        }
    }

    public void testNoMatchingSegmentGivesNoSlices() throws IOException {
        try (Directory directory = newDirectory()) {
            index(directory);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                assertThat(slices(reader, aEquals(1_000), 4, true), hasSize(0));
            }
        }
    }

    public void testQueryWithoutSortKeyFilterIsSlicedAsBefore() throws IOException {
        try (Directory directory = newDirectory()) {
            index(directory);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                Query query = new BooleanQuery.Builder().add(
                    SortedNumericDocValuesField.newSlowRangeQuery("other", 1, 2),
                    BooleanClause.Occur.FILTER
                ).build();
                assertThat(slices(reader, query, 4, true).size(), equalTo(slices(reader, query, 4, false).size()));
            }
        }
    }
}
