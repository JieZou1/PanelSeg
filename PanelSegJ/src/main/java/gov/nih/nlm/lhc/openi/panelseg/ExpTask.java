package gov.nih.nlm.lhc.openi.panelseg;

import java.util.concurrent.RecursiveAction;

/**
 * Implementing multi-threading processing in Fork/Join framework for Exp derived classes
 * It takes an Exp, and uses divide-and-conquer strategy to run multi-tasks in commonPool
 *
 * Created by jzou on 9/12/2016.
 */
class ExpTask extends RecursiveAction
{
    private static final long serialVersionUID = 1L;

    int seqThreshold;

    Exp seqTrain;	int start, end;

    ExpTask(Exp segTrain, int start, int end, int seqThreshold)
    {
        this.seqTrain = segTrain;		this.start = start;		this.end = end; this.seqThreshold = seqThreshold;
    }

    @Override
    protected void compute()
    {
        if (end - start < seqThreshold)
        {
            for (int i = start; i < end; i++)
            {
                try {
                    seqTrain.generate(i);
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
        else
        {
            int middle = (start + end)/2;
            invokeAll(	new ExpTask(seqTrain, start, middle, this.seqThreshold),
                    new ExpTask(seqTrain, middle, end, this.seqThreshold));
        }
    }

}
