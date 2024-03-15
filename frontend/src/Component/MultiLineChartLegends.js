import React, { Component } from 'react';


class MultiLineChartLegends extends Component {
    constructor(props) {
        super(props);
        this.state = {
        }
    }

    componentDidMount() {
        // this.getCardsData()
    }

    render() {
        const { colors, i, selectedTrend, data } = this.props;
        return (
            i ?
                (<div className='legends-box' style={{ width: '250px' }}>
                    <div className="triangle-up"></div>
                    <div id="legend">
                        {data.map((d, i) =>
                            <p className="sub-legend"><div className="square" style={{ background: colors[i] }}></div>{d}</p>
                        )}
                        <p className="sub-legend">{`X-axis : ${selectedTrend} numbers`}</p>
                    </div>
                </div>) :

                (<div className="bottom-legends">
                    {data.map((d, i) =>
                        <p className="sub-legend"><div className="square" style={{ background: colors[i] }}></div>{d}</p>
                    )}
                    <p className="sub-legend">{`X-axis : ${selectedTrend} numbers`}</p>
                </div>)
        );
    }
}
export default MultiLineChartLegends;




