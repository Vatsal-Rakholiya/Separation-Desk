export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type JsonInit = RequestInit & { json?: unknown };

export async function api<T>(path: string, options: JsonInit = {}): Promise<T> {
  const headers = new Headers(options.headers ?? {});
  const init: RequestInit = { ...options, headers };

  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    init.body = JSON.stringify(options.json);
  }

  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("Request failed. Check that the Flask server is running.");
  }

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "string"
        ? payload
        : (payload as { error?: string }).error ?? "Request failed.";
    throw new ApiError(message, response.status);
  }

  return payload as T;
}

export async function uploadFiles<T>(files: FileList | File[]): Promise<T> {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "string"
        ? payload
        : (payload as { error?: string }).error ?? "Upload failed.";
    throw new ApiError(message, response.status);
  }

  return payload as T;
}
