/**
 * Said when `withholdSecrets` replaced something: the API sent a value shaped
 * like a credential, which is an API bug. Names where, never what.
 */
export function WithheldNotice({ paths }: { paths: string[] }) {
  if (paths.length === 0) return null;
  return (
    <p className="callout error" role="alert" data-testid="withheld-notice">
      The API returned {paths.length === 1 ? "a value" : `${paths.length} values`} that looked like a
      credential. {paths.length === 1 ? "It was" : "They were"} not shown and appear as{" "}
      <code>[withheld]</code>. This is an API bug — please report it (fields:{" "}
      <code>{paths.join(", ")}</code>).
    </p>
  );
}
