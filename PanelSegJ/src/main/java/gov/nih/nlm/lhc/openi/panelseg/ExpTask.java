package gov.nih.nlm.lhc.openi.panelseg;

import java.util.concurrent.RecursiveAction;

/**
 * Implementing multi-threading processing in Fork/Join framework for Exp derived classes
 *
 * Created by jzou on 9/12/2016.
 */
class ExpTask extends RecursiveAction
{
    private static final long serialVersionUID = 1L;

    private Exp exp;
    private int start, end;

    ExpTask(Exp exp, int start, int end)
    {
        this.exp = exp;		this.start = start;		this.end = end;
        //this.cores = Runtime.getRuntime().availableProcessors() - 2; //# of cores to use. We left at least one core for other apps.
    }

    @Override
    protected void compute()
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

    /**
     * Generate all ExpTasks.
     * @param exp   An Exp
     * @param nTotal    total number of samples to be processed.
     * @param nThreads  total number of threads to be used.
     * @return
     */
    static ExpTask[] createTasks(Exp exp, int nTotal, int nThreads)
    {
        int stride = (nTotal % nThreads > 0)? nTotal / (nThreads - 1): nTotal / nThreads;

        ExpTask[] tasks = new ExpTask[nThreads]; int starts[] = new int[nThreads], ends[] = new int[nThreads];

        for (int i = 0; i < nThreads; i++)
        {
            starts[i] = i * stride;
            ends[i] = (i + 1) * stride; if (ends[i] > exp.imagePaths.size()) ends[i] = exp.imagePaths.size();

            tasks[i] = new ExpTask(exp, starts[i], ends[i]);
        }

        return tasks;
    }
}
