jq --slurp '
  def score: 
    map(
      . + {score: (((.best_loss | log10) + (.mean_loss | log10)? // (.best_loss | log10) * 2.2395572632422884) * (1.0 - .best_accuracy))}
    ) | 
    sort_by(.score) | 
    .[0:5];
  
  {
    <model>: (.[0] | score)
  }                              
' <model>.json
