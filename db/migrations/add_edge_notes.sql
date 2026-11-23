-- Keep the mark the board puts on a pick that is not an ordinary one.
--
-- build_prop_edges sets `notes` on two kinds of row: `under_even`, a pick the
-- corrected chance puts at even or worse, and from 22 November 2026
-- `against_number`, a pick on the side the corrected chance favours where our
-- own number sits on the other side of the line. Both are published as small
-- and kept out of the gap ranking.
--
-- The builder writes only the columns the table has and drops the rest with a
-- warning, so without this the mark never left the builder's memory and the
-- record had no way to grade these picks on their own, which is the reason the
-- mark exists.
--
-- prop_edges_history as well: the archive step inserts using prop_edges'
-- column list, so a column missing there fails the whole build.
ALTER TABLE prop_edges ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE prop_edges_history ADD COLUMN IF NOT EXISTS notes TEXT;
