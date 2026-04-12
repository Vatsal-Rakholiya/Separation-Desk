export type MatchMode = "exact" | "nearest" | "spread";
export type SelectionMode = "keep" | "remove";
export type CompressionMode = "lzw" | "deflate" | "none";
export type PreviewQuality = "proxy" | "full";

export interface Recipe {
  name: string;
  cmyk: [number, number, number, number];
}

export interface ManualRecipe {
  c: number;
  m: number;
  y: number;
  k: number;
}

export interface Settings {
  match_mode: MatchMode;
  selection_mode: SelectionMode;
  spread: number;
  invert_grayscale: boolean;
  gray_gamma: number;
  compression: CompressionMode;
  preview_quality: PreviewQuality;
  preview_sharpen: boolean;
  operator: string;
  client: string;
  revision: string;
  notes: string;
}

export interface JobPage {
  index: number;
  label: string;
  shape: number[];
  dpi_x: number;
  dpi_y: number;
  photometric: string;
  compression: string;
  dtype: string;
}

export interface ResultStats {
  matched_pixels?: number;
  matched_percent?: number;
  processing_seconds: number;
  ink_coverage?: number;
  warnings?: string[];
  width?: number;
  height?: number;
  dpi_x?: number;
  dpi_y?: number;
}

export interface ResultSummary {
  result_id: string;
  operation?: SelectionMode;
  status: string;
  file_base?: string;
  stats: ResultStats;
  previews: Record<string, string>;
  downloads: {
    zip: string;
  };
}

export interface SelectedRunSummary {
  result_id: string;
  status: string;
  stats: ResultStats;
  previews: {
    original: string;
  };
  downloads: {
    zip: string;
  };
  selected_result: ResultSummary;
  removed_result: ResultSummary;
}

export interface HistoryEntry {
  id: string;
  time: string;
  job_name: string;
  file_name: string;
  page: number;
  matched_pixels: number;
  matched_percent: number;
  processing_seconds: number;
  kind: "selected" | "standard";
  summary: ResultSummary | SelectedRunSummary;
}

export interface Job {
  id: string;
  file_name: string;
  job_name: string;
  signature: string;
  pages: JobPage[];
  current_page: number;
  selected_colors: Recipe[];
  last_result: ResultSummary | SelectedRunSummary | null;
  last_standard_result: ResultSummary | null;
}

export interface AppState {
  jobs: Job[];
  active_job_id: string | null;
  recent_colors: Recipe[];
  favorites: Recipe[];
  profiles: string[];
  history: HistoryEntry[];
  settings: Settings;
  presets: Recipe[];
  click_pick_available: boolean;
}

export interface ProfilePayload {
  settings?: Settings;
  selected_colors?: Recipe[];
  manual_recipe?: ManualRecipe;
}

export interface SampleMatch {
  cmyk: [number, number, number, number];
  pixels: number;
  distance: number;
}

export interface SampleResponse {
  x: number;
  y: number;
  cmyk: [number, number, number, number];
  hex: string;
  nearest: SampleMatch[];
}

export interface InspectorPreviewPayload {
  width: number;
  height: number;
  source_width: number;
  source_height: number;
  data: string;
}
