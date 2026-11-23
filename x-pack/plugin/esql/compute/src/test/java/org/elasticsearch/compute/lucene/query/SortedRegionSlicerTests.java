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
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.IndexSortSortedNumericDocValuesRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.store.Directory;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.compute.lucene.PartialLeafReaderContext;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;

/**
 * Tests {@link SortedRegionSlicer} against real Lucene indices with an index sort and several segments. The regions are
 * compared with a brute force evaluation of the sort key filters.
 */
public class SortedRegionSlicerTests extends ESTestCase {
    /**
     * A doc of the test index. {@code c} is not part of the index sort.
     */
    private record TestDoc(long a, long b, long c) {}

    private static Query range(String field, long lower, long upper) {
        // Same shape as the query ES builds for a range on a field of an index sorted on that field.
        return new IndexSortSortedNumericDocValuesRangeQuery(
            field,
            lower,
            upper,
            SortedNumericDocValuesField.newSlowRangeQuery(field, lower, upper)
        );
    }

    private static Query filter(Query... queries) {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (Query query : queries) {
            builder.add(query, BooleanClause.Occur.FILTER);
        }
        return new ConstantScoreQuery(builder.build());
    }

    private static Document document(TestDoc doc) {
        Document document = new Document();
        document.add(new SortedNumericDocValuesField("a", doc.a));
        document.add(new SortedNumericDocValuesField("b", doc.b));
        document.add(new SortedNumericDocValuesField("c", doc.c));
        return document;
    }

    private static Sort sort(SortField.Type type, boolean reverseA, boolean reverseB) {
        return new Sort(new SortedNumericSortField("a", type, reverseA), new SortedNumericSortField("b", type, reverseB));
    }

    /**
     * Indexes every list of docs as one segment. Merges are disabled so the segments stay as they are flushed.
     */
    private static IndexWriter writer(Directory directory, Sort sort) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig();
        config.setMergePolicy(NoMergePolicy.INSTANCE);
        if (sort != null) {
            config.setIndexSort(sort);
        }
        return new IndexWriter(directory, config);
    }

    private static void addSegment(IndexWriter writer, List<TestDoc> docs) throws IOException {
        for (TestDoc doc : docs) {
            writer.addDocument(document(doc));
        }
        writer.commit();
    }

    private static Set<Integer> matches(IndexSearcher searcher, Query query) throws IOException {
        Set<Integer> docs = new TreeSet<>();
        for (ScoreDoc scoreDoc : searcher.search(query, Math.max(1, searcher.getIndexReader().maxDoc())).scoreDocs) {
            docs.add(scoreDoc.doc);
        }
        return docs;
    }

    /**
     * The global doc ids covered by {@code groups}. Fails if a doc is covered twice or a slice has more than one leaf.
     */
    private static Set<Integer> covered(List<List<PartialLeafReaderContext>> groups) {
        Set<Integer> docs = new TreeSet<>();
        for (List<PartialLeafReaderContext> group : groups) {
            assertThat(group, hasSize(1));
            PartialLeafReaderContext leaf = group.get(0);
            assertTrue("empty slice " + leaf, leaf.minDoc() < leaf.maxDoc());
            assertThat(leaf.maxDoc() - leaf.minDoc(), lessThanOrEqualTo(LuceneSliceQueue.MAX_DOCS_PER_SLICE));
            for (int doc = leaf.minDoc(); doc < leaf.maxDoc(); doc++) {
                assertTrue("doc covered twice " + doc, docs.add(leaf.leafReaderContext().docBase + doc));
            }
        }
        return docs;
    }

    private static List<TestDoc> randomSegment(int size, long baseA, int valuesA, long baseB, int valuesB) {
        List<TestDoc> docs = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            docs.add(new TestDoc(baseA + randomInt(valuesA - 1), baseB + randomInt(valuesB - 1), randomInt(100)));
        }
        return docs;
    }

    /**
     * Random indices, sort directions and filters. When the optimization applies, the sliced docs must be exactly the docs that match
     * the sort key part of the filter, and every doc that matches the whole query must be sliced.
     */
    public void testRandomRegionsMatchBruteForce() throws IOException {
        int valuesA = 24;
        int valuesB = 12;
        for (int iteration = 0; iteration < 30; iteration++) {
            long baseA = randomFrom(Long.MIN_VALUE, -1_000_000L, -3L, 0L, 17L, Long.MAX_VALUE - valuesA - 10);
            long baseB = randomFrom(Long.MIN_VALUE, -5L, 0L, 99L, Long.MAX_VALUE - valuesB - 10);
            Sort sort = sort(SortField.Type.LONG, randomBoolean(), randomBoolean());
            try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort)) {
                int segments = between(1, 5);
                for (int s = 0; s < segments; s++) {
                    addSegment(writer, randomSegment(between(50, 600), baseA, valuesA, baseB, valuesB));
                }
                try (DirectoryReader reader = DirectoryReader.open(writer)) {
                    assertThat(reader.leaves(), hasSize(segments));
                    IndexSearcher searcher = new IndexSearcher(reader);

                    long lowerA = baseA + between(0, valuesA - 1);
                    long upperA = randomBoolean() ? lowerA : lowerA + between(0, 3);
                    if (randomInt(9) == 0) {
                        lowerA = Long.MIN_VALUE;
                    }
                    if (randomInt(9) == 0) {
                        upperA = Long.MAX_VALUE;
                    }
                    List<Query> keyClauses = new ArrayList<>();
                    keyClauses.add(range("a", lowerA, upperA));
                    boolean useB = randomBoolean();
                    long lowerB = baseB + between(0, valuesB - 1);
                    long upperB = lowerB + between(0, 4);
                    if (useB) {
                        keyClauses.add(range("b", lowerB, upperB));
                    }
                    List<Query> allClauses = new ArrayList<>(keyClauses);
                    if (randomBoolean()) {
                        // Not a sort key: must be ignored by the region search and must not break it.
                        allClauses.add(range("c", 0, between(0, 100)));
                    }
                    Query query = filter(allClauses.toArray(Query[]::new));

                    // The second sort key only takes part when the first one is pinned to a single value.
                    boolean secondKeyUsed = useB && lowerA == upperA;
                    Query regionQuery = secondKeyUsed
                        ? filter(range("a", lowerA, upperA), range("b", lowerB, upperB))
                        : filter(range("a", lowerA, upperA));
                    Set<Integer> expectedRegion = matches(searcher, regionQuery);

                    int taskConcurrency = between(1, 12);
                    List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(searcher, query, taskConcurrency);

                    boolean expectNarrowed = expectedRegion.size() <= reader.maxDoc() * SortedRegionSlicer.MAX_REGION_FRACTION;
                    if (expectNarrowed == false) {
                        assertThat("region is too big to be worth it", groups, nullValue());
                        continue;
                    }
                    assertThat(groups, notNullValue());
                    Set<Integer> sliced = covered(groups);
                    assertThat(sliced, equalTo(expectedRegion));
                    assertTrue("every match is in a slice", sliced.containsAll(matches(searcher, query)));
                    assertThat(groups.size(), lessThanOrEqualTo(taskConcurrency + segments));
                }
            }
        }
    }

    public void testSlicesOnlyTheRegionOfOneValue() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            List<TestDoc> docs = new ArrayList<>();
            for (int i = 0; i < 10_000; i++) {
                docs.add(new TestDoc(i / 100, randomInt(50), 0));
            }
            addSegment(writer, docs);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                // a == 42 is docs [4200, 4300): 100 docs, below the slice floor, so one slice
                List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(
                    new IndexSearcher(reader),
                    filter(range("a", 42, 42)),
                    4
                );
                assertThat(groups, hasSize(1));
                assertThat(groups.get(0).get(0).minDoc(), equalTo(4200));
                assertThat(groups.get(0).get(0).maxDoc(), equalTo(4300));
            }
        }
    }

    public void testRegionIsCutIntoTaskConcurrencySlices() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            List<TestDoc> docs = new ArrayList<>();
            for (int i = 0; i < 40_000; i++) {
                docs.add(new TestDoc(i / 4_000, randomInt(50), 0));
            }
            addSegment(writer, docs);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                // a == 3 is docs [12000, 16000), 4000 docs: 4 slices of 1000 for a concurrency of 4
                List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(
                    new IndexSearcher(reader),
                    filter(range("a", 3, 3)),
                    4
                );
                assertThat(groups, hasSize(4));
                for (int i = 0; i < 4; i++) {
                    assertThat(groups.get(i).get(0).minDoc(), equalTo(12_000 + i * 1000));
                    assertThat(groups.get(i).get(0).maxDoc(), equalTo(12_000 + (i + 1) * 1000));
                }
                // a bigger concurrency does not cut below the 1000 docs floor
                groups = SortedRegionSlicer.slice(new IndexSearcher(reader), filter(range("a", 3, 3)), 100);
                assertThat(groups, hasSize(4));
            }
        }
    }

    public void testSecondSortKeyNarrowsTheRegionWhenFirstIsPinned() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, true, true))) {
            List<TestDoc> docs = new ArrayList<>();
            for (int i = 0; i < 20_000; i++) {
                docs.add(new TestDoc(i % 10, (i / 10) % 100, 0));
            }
            addSegment(writer, docs);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                // 2000 docs have a == 2 and half of them have a b in [10, 59]
                Query pinned = filter(range("a", 2, 2), range("b", 10, 59));
                assertThat(covered(SortedRegionSlicer.slice(searcher, pinned, 3)), equalTo(matches(searcher, pinned)));
                // a range on the first key ignores the second key, so the region is everything with a in [1, 2]: 4000 docs
                Query wide = filter(range("a", 1, 2), range("b", 10, 59));
                assertThat(covered(SortedRegionSlicer.slice(searcher, wide, 3)), equalTo(matches(searcher, filter(range("a", 1, 2)))));
            }
        }
    }

    public void testPrunesSegmentsWithoutTheValue() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, true, false))) {
            addSegment(writer, randomSegment(5_000, 100, 10, 0, 10));
            addSegment(writer, randomSegment(5_000, 0, 10, 0, 10));
            List<TestDoc> withValue = randomSegment(5_000, 0, 10, 0, 10);
            withValue.addAll(randomSegment(2_000, 1_000, 1, 0, 10));
            addSegment(writer, withValue);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                assertThat(reader.leaves(), hasSize(3));
                IndexSearcher searcher = new IndexSearcher(reader);
                Query query = filter(range("a", 1_000, 1_000));
                List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(searcher, query, 4);
                assertThat(groups, notNullValue());
                assertThat(covered(groups), equalTo(matches(searcher, query)));
                for (List<PartialLeafReaderContext> group : groups) {
                    assertThat(group.get(0).leafReaderContext().ord, equalTo(2));
                }
            }
        }
    }

    public void testNoDocCanMatch() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            addSegment(writer, randomSegment(5_000, 0, 10, 0, 10));
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                // Narrowed to nothing is not the same as not applicable: there is nothing to scan.
                assertThat(SortedRegionSlicer.slice(new IndexSearcher(reader), filter(range("a", 500, 600)), 4), empty());
                assertThat(SortedRegionSlicer.slice(new IndexSearcher(reader), filter(range("a", 7, 3)), 4), empty());
            }
        }
    }

    public void testIntSortField() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.INT, false, true))) {
            List<TestDoc> docs = new ArrayList<>();
            for (int i = 0; i < 20_000; i++) {
                docs.add(new TestDoc(i / 1_000, randomInt(30), 0));
            }
            addSegment(writer, docs);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                Query query = filter(range("a", 5, 5));
                assertThat(covered(SortedRegionSlicer.slice(searcher, query, 2)), equalTo(matches(searcher, query)));
            }
        }
    }

    public void testNotApplicable() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            addSegment(writer, randomSegment(5_000, 0, 20, 0, 10));
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                // no filter on a sort key
                assertThat(SortedRegionSlicer.slice(searcher, Queries.ALL_DOCS_INSTANCE, 4), nullValue());
                assertThat(SortedRegionSlicer.slice(searcher, filter(range("c", 0, 10)), 4), nullValue());
                // a filter on the second sort key only
                assertThat(SortedRegionSlicer.slice(searcher, filter(range("b", 1, 1)), 4), nullValue());
                // the range is under a SHOULD, a doc can match without it
                BooleanQuery.Builder should = new BooleanQuery.Builder();
                should.add(range("a", 3, 3), BooleanClause.Occur.SHOULD);
                should.add(range("c", 0, 10), BooleanClause.Occur.SHOULD);
                assertThat(SortedRegionSlicer.slice(searcher, should.build(), 4), nullValue());
                // too much of the shard: the regular partitioner is a good fit
                assertThat(SortedRegionSlicer.slice(searcher, filter(range("a", 0, 14)), 4), nullValue());
            }
        }
    }

    public void testSegmentWithoutIndexSortIsNotProbed() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, null)) {
            addSegment(writer, randomSegment(5_000, 0, 20, 0, 10));
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                assertThat(SortedRegionSlicer.slice(new IndexSearcher(reader), filter(range("a", 3, 3)), 4), nullValue());
            }
        }
    }

    public void testSegmentWithDeletedDocsIsKeptWhole() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            addSegment(writer, randomSegment(300, 0, 20, 0, 10));
            writer.deleteDocuments(SortedNumericDocValuesField.newSlowExactQuery("a", 3));
            writer.commit();
            addSegment(writer, randomSegment(10_000, 0, 20, 0, 10));
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                assertThat(reader.leaves(), hasSize(2));
                assertTrue(reader.leaves().get(0).reader().hasDeletions());
                IndexSearcher searcher = new IndexSearcher(reader);
                Query query = filter(range("a", 3, 3));
                List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(searcher, query, 4);
                assertThat(groups, notNullValue());
                Set<Integer> sliced = covered(groups);
                assertTrue("the segment with deletions is not narrowed", sliced.containsAll(matches(searcher, query)));
                LeafReaderContext deleted = reader.leaves().get(0);
                for (int doc = 0; doc < deleted.reader().maxDoc(); doc++) {
                    assertTrue("whole segment with deletions " + doc, sliced.contains(deleted.docBase + doc));
                }
                // the segment without deletions was narrowed
                LeafReaderContext clean = reader.leaves().get(1);
                assertThat(
                    sliced.stream().filter(doc -> doc >= clean.docBase).count(),
                    equalTo(matches(searcher, query).stream().filter(doc -> doc >= clean.docBase).count())
                );
            }
        }
    }

    public void testSparseAndMultiValuedFieldsAreNotProbed() throws IOException {
        // a doc without a value for the sort field
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            List<TestDoc> docs = randomSegment(5_000, 0, 20, 0, 10);
            addSegment(writer, docs);
            Document sparse = new Document();
            sparse.add(new SortedNumericDocValuesField("c", 1));
            writer.addDocument(sparse);
            writer.commit();
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                // The first segment is probed, the sparse segment is kept whole. Nothing is lost.
                Query query = filter(range("a", 3, 3));
                List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(searcher, query, 4);
                assertThat(groups, notNullValue());
                assertTrue(covered(groups).containsAll(matches(searcher, query)));
            }
        }
        // a doc with two values for the sort field
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            List<Document> docs = new ArrayList<>();
            for (TestDoc doc : randomSegment(5_000, 0, 20, 0, 10)) {
                docs.add(document(doc));
            }
            docs.get(0).add(new SortedNumericDocValuesField("a", 19));
            for (Document doc : docs) {
                writer.addDocument(doc);
            }
            writer.commit();
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                assertThat(SortedRegionSlicer.slice(new IndexSearcher(reader), filter(range("a", 3, 3)), 4), nullValue());
            }
        }
    }

    public void testGroupsAreInDocOrderWithinASegment() throws IOException {
        try (Directory directory = newDirectory(); IndexWriter writer = writer(directory, sort(SortField.Type.LONG, false, false))) {
            List<TestDoc> docs = new ArrayList<>();
            for (int i = 0; i < 30_000; i++) {
                docs.add(new TestDoc(i / 3_000, 0, 0));
            }
            addSegment(writer, docs);
            try (DirectoryReader reader = DirectoryReader.open(writer)) {
                List<List<PartialLeafReaderContext>> groups = SortedRegionSlicer.slice(
                    new IndexSearcher(reader),
                    filter(range("a", 4, 4)),
                    3
                );
                List<Integer> starts = new ArrayList<>();
                for (List<PartialLeafReaderContext> group : groups) {
                    starts.add(group.get(0).minDoc());
                }
                assertThat(starts, contains(12_000, 13_000, 14_000));
            }
        }
    }
}
