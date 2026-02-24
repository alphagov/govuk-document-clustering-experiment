COPY (
  SELECT
    id,
    title,
    details->'body' as body
  FROM
    content_items
  WHERE
    id IN (SELECT id FROM content_item_and_taxons WHERE taxon_base_path = '/employment/labour-market-reform')
) to STDOUT WITH CSV HEADER;
