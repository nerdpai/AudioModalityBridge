jq --slurp '
  def score: 
    map(
      . + {score: (1.0 - .mean_accuracy)}
    ) | 
    sort_by(.score) | 
    .[0:5];
  
  {
    <model>: (.[0] | score)
  }                              
' <model>.json
