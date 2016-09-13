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

    Exp exp;	int start, end;

    ExpTask(Exp exp, int start, int end, int seqThreshold)
    {
        this.exp = exp;		this.start = start;		this.end = end; this.seqThreshold = seqThreshold;
    }

    ExpTask(Exp segTrain, int start, int end)
    {
        this.exp = segTrain;		this.start = start;		this.end = end;
    }

    @Override
    protected void compute()
    {
        if (end - start < exp.imagePaths.size())
        {
            for (int i = start; i < end; i++)
            {
                try {
                    exp.generate(i);
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
        }
        else
        {
            int n = 15; //# of cores to use.
            int k = (exp.imagePaths.size() % n > 0) ? n+1: n; //# of threads to create
            ExpTask[] tasks = new ExpTask[k]; int starts[] = new int[k], ends[] = new int[k];

            int stride = exp.imagePaths.size() / n;
            for (int i = 0; i < k; i++)
            {
                starts[i] = i * stride;
                ends[i] = (i + 1) * stride; if (ends[i] > exp.imagePaths.size()) ends[i] = exp.imagePaths.size();

                tasks[i] = new ExpTask(exp, starts[i], ends[i]);
            }
            invokeAll(tasks);

        }
    }

}
