import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'
import DownloadCSV from '../../Component/DownloadCSV';

class RPSPosDownloadCustomHeaderGroup extends Component {
    constructor(props) {
        super(props);
        this.state = {
            chartVisible: false
        }
    }

    closeChartModal() {
        this.setState({ chartVisible: false })
    }

    render() {
        const { displayName } = this.props;
        const downloadURL = localStorage.getItem('RPSPOSDownloadURL')
        return (
            <div>
                <div className="ag-header-group-cell-label">
                    <div className="customHeaderLabel">{displayName}</div>
                    <DownloadCSV url={downloadURL} name={`RPS POS`} path={`/rpsPos`} page={`RPSPOS Page`} />
                </div>

            </div>
        );
    }

}

export default RPSPosDownloadCustomHeaderGroup;