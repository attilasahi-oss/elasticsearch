/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute.lucene.query;

import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.NumericDocValuesRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.elasticsearch.compute.lucene.PartialLeafReaderContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Cuts slices only from the doc id region that can match a query with range filters on the leading index sort fields.
 *
 * <p>A segment sorted on {@code (a, b, ...)} stores all docs with {@code a == x} next to each other, and inside that run all docs
 * with {@code b} in a range are next to each other too. A query like {@code a == 62 AND b >= 100 AND b <= 200 AND other filters}
 * therefore matches only inside one contiguous run of every segment. Slicing by doc count alone ignores that: the matching docs
 * can sit in a handful of slices, so most drivers get nothing to do. Here the run is located with a binary search over the doc
 * values, segments that cannot match are dropped, and the run is cut into about {@code taskConcurrency} slices.
 *
 * <p>The slices only restrict which doc ids are visited. Every slice still evaluates the full query, so a region that is
 * larger than the exact set of matching docs only costs speed, never correctness.
 *
 * <p>The search needs to read the sort fields by doc id, so a segment is only probed when all of the following hold. Any other
 * segment is kept whole instead:
 * <ul>
 *     <li>the segment records an index sort whose first field is a long or int sort,</li>
 *     <li>the segment has no deleted docs and no nested doc blocks (the sort only orders the root docs of a block),</li>
 *     <li>the constrained sort fields are single valued and have a value in every doc.</li>
 * </ul>
 * The optimization is skipped for the whole query unless it narrowed at least one segment and the remaining docs are a small
 * part of the shard, because then the regular partitioner is already a good fit.
 */
final class SortedRegionSlicer {
    /**
     * Smallest slice we cut from a region. Below this the per slice setup cost dominates the scan.
     */
    static final int MIN_SLICE_DOCS = 1_000;

    /**
     * The optimization only applies when the regions are at most this fraction of the shard's docs.
     */
    static final double MAX_REGION_FRACTION = 0.25;

    private SortedRegionSlicer() {}

    /**
     * @return the slice groups, or {@code null} when the optimization does not apply and the caller must use the regular partitioner
     */
    static List<List<PartialLeafReaderContext>> slice(IndexSearcher searcher, Query query, int taskConcurrency) throws IOException {
        List<NumericDocValuesRangeQuery> ranges = new ArrayList<>();
        collectRanges(query, ranges);
        if (ranges.isEmpty()) {
            return null;
        }
        List<PartialLeafReaderContext> regions = new ArrayList<>();
        long regionDocs = 0;
        boolean narrowed = false;
        for (LeafReaderContext leaf : searcher.getLeafContexts()) {
            PartialLeafReaderContext region = probe(leaf, ranges);
            if (region == null) {
                // Cannot probe this segment, so any of its docs may match.
                region = new PartialLeafReaderContext(leaf);
            } else {
                narrowed = true;
            }
            if (region.minDoc() < region.maxDoc()) {
                regions.add(region);
                regionDocs += region.maxDoc() - region.minDoc();
            }
        }
        if (narrowed == false || regionDocs > searcher.getIndexReader().maxDoc() * MAX_REGION_FRACTION) {
            return null;
        }
        long sliceSize = Math.min(
            LuceneSliceQueue.MAX_DOCS_PER_SLICE,
            Math.max(MIN_SLICE_DOCS, Math.ceilDiv(regionDocs, Math.max(1, taskConcurrency)))
        );
        List<List<PartialLeafReaderContext>> groups = new ArrayList<>();
        for (PartialLeafReaderContext region : regions) {
            int width = region.maxDoc() - region.minDoc();
            int pieces = (int) Math.ceilDiv(width, sliceSize);
            int base = width / pieces;
            int extra = width % pieces;
            int minDoc = region.minDoc();
            for (int i = 0; i < pieces; i++) {
                int maxDoc = minDoc + base + (i < extra ? 1 : 0);
                groups.add(List.of(new PartialLeafReaderContext(region.leafReaderContext(), minDoc, maxDoc)));
                minDoc = maxDoc;
            }
        }
        return groups;
    }

    /**
     * Collects the numeric range clauses that every matching doc must satisfy. Clauses under a {@code SHOULD} or {@code MUST_NOT}
     * occurrence are skipped. Skipping a clause only makes the region larger.
     */
    private static void collectRanges(Query query, List<NumericDocValuesRangeQuery> out) {
        if (query instanceof NumericDocValuesRangeQuery range) {
            out.add(range);
        } else if (query instanceof ConstantScoreQuery constantScore) {
            collectRanges(constantScore.getQuery(), out);
        } else if (query instanceof BoostQuery boost) {
            collectRanges(boost.getQuery(), out);
        } else if (query instanceof BooleanQuery bool) {
            for (BooleanClause clause : bool.clauses()) {
                if (clause.occur() == BooleanClause.Occur.FILTER || clause.occur() == BooleanClause.Occur.MUST) {
                    collectRanges(clause.query(), out);
                }
            }
        }
    }

    /**
     * @return the doc id region of {@code leaf} that can match, or {@code null} if the leaf cannot be probed. An empty region
     *         means that no doc of the leaf can match.
     */
    private static PartialLeafReaderContext probe(LeafReaderContext leaf, List<NumericDocValuesRangeQuery> ranges) throws IOException {
        LeafReader reader = leaf.reader();
        Sort sort = reader.getMetaData().sort();
        if (sort == null || reader.hasDeletions() || reader.getMetaData().hasBlocks() || reader.maxDoc() == 0) {
            return null;
        }
        SortField[] sortFields = sort.getSort();
        if (isIntegerSort(sortFields[0]) == false) {
            return null;
        }
        Bounds first = Bounds.of(ranges, sortFields[0].getField());
        if (first == null) {
            return null;
        }
        // The second sort field only narrows the region when the first one is pinned to a single value.
        Bounds second = null;
        SortField secondSort = null;
        if (sortFields.length > 1 && first.lower == first.upper && isIntegerSort(sortFields[1])) {
            second = Bounds.of(ranges, sortFields[1].getField());
            secondSort = second == null ? null : sortFields[1];
        }
        SortKeys keys = new SortKeys(reader, sortFields[0], first, secondSort, second);
        if (keys.usable() == false) {
            return null;
        }
        int start = keys.firstDocNotBefore(0, reader.maxDoc(), false);
        int end = keys.firstDocNotBefore(start, reader.maxDoc(), true);
        return new PartialLeafReaderContext(leaf, start, end);
    }

    private static boolean isIntegerSort(SortField sortField) {
        return sortField instanceof SortedNumericSortField numeric
            && (numeric.getNumericType() == SortField.Type.LONG || numeric.getNumericType() == SortField.Type.INT);
    }

    /**
     * Inclusive bounds on one field, the intersection of all range clauses on it.
     */
    private static final class Bounds {
        private long lower = Long.MIN_VALUE;
        private long upper = Long.MAX_VALUE;

        /**
         * @return the bounds for {@code field} or {@code null} if no clause constrains it
         */
        static Bounds of(List<NumericDocValuesRangeQuery> ranges, String field) {
            Bounds bounds = null;
            for (NumericDocValuesRangeQuery range : ranges) {
                if (range.getField().equals(field)) {
                    if (bounds == null) {
                        bounds = new Bounds();
                    }
                    bounds.lower = Math.max(bounds.lower, range.lowerValue());
                    bounds.upper = Math.min(bounds.upper, range.upperValue());
                }
            }
            return bounds;
        }
    }

    /**
     * Random access to the sort keys of one segment, ordered like the segment's index sort. The box described by the bounds is a
     * contiguous run of docs in that order: {@code [firstDocNotBefore(false), firstDocNotBefore(true))}.
     */
    private static final class SortKeys {
        private final LeafReader reader;
        private final String field1;
        private final boolean reverse1;
        private final Bounds bounds1;
        private final String field2;
        private final boolean reverse2;
        private final Bounds bounds2;

        SortKeys(LeafReader reader, SortField first, Bounds bounds1, SortField second, Bounds bounds2) {
            this.reader = reader;
            this.field1 = first.getField();
            this.reverse1 = first.getReverse();
            this.bounds1 = bounds1;
            this.field2 = second == null ? null : second.getField();
            this.reverse2 = second != null && second.getReverse();
            this.bounds2 = bounds2;
        }

        boolean usable() throws IOException {
            return isDenseAndSingleValued(field1) && (field2 == null || isDenseAndSingleValued(field2));
        }

        private boolean isDenseAndSingleValued(String field) throws IOException {
            NumericDocValues values = singleValued(field);
            return values != null && values.cost() == reader.maxDoc();
        }

        private NumericDocValues singleValued(String field) throws IOException {
            SortedNumericDocValues values = DocValues.getSortedNumeric(reader, field);
            return DocValues.unwrapSingleton(values);
        }

        private long value(String field, int doc) throws IOException {
            // A fresh iterator per probe: doc values iterators only move forward and the binary search jumps back and forth.
            NumericDocValues values = singleValued(field);
            if (values == null || values.advanceExact(doc) == false) {
                throw new IllegalStateException("expected a dense single valued field [" + field + "] for doc [" + doc + "]");
            }
            return values.longValue();
        }

        /**
         * Binary search for the first doc in {@code [from, to)} that is not sorted before the box ({@code pastBox == false}) or that
         * is sorted after the box ({@code pastBox == true}); {@code to} if there is none.
         */
        int firstDocNotBefore(int from, int to, boolean pastBox) throws IOException {
            int low = from;
            int high = to;
            while (low < high) {
                int mid = (low + high) >>> 1;
                if (isBefore(mid, pastBox)) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            return low;
        }

        /**
         * With {@code pastBox == false}: is the doc sorted strictly before the first position of the box? With
         * {@code pastBox == true}: is it sorted before or inside the box, that is not after its last position?
         */
        private boolean isBefore(int doc, boolean pastBox) throws IOException {
            int comparison = compareToEdge(field1, reverse1, bounds1, doc, pastBox);
            if (comparison == 0 && field2 != null) {
                comparison = compareToEdge(field2, reverse2, bounds2, doc, pastBox);
            }
            return pastBox ? comparison <= 0 : comparison < 0;
        }

        /**
         * Compares the doc's value to one edge of the bounds in sort order: negative if the doc sorts before the edge, zero if it is
         * on it and positive if it sorts after it. The edge is the first position of the range when {@code pastBox == false} and
         * the last position when it is {@code true}.
         */
        private int compareToEdge(String field, boolean reverse, Bounds bounds, int doc, boolean pastBox) throws IOException {
            long value = value(field, doc);
            long edge = (pastBox ^ reverse) ? bounds.upper : bounds.lower;
            return reverse ? Long.compare(edge, value) : Long.compare(value, edge);
        }
    }
}
